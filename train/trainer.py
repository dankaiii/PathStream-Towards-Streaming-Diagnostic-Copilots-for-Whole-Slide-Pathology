"""Train one stage and select the checkpoint with minimum validation loss."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from data.collator import FrameMatches, SessionCollator
from data.conversation import make_processor
from data.dataset import SessionDataset
from data.sampler import ShuffledEpochSampler, validation_partition
from losses.diagnosis_attention import ATTENTION_GROUPS
from model.qwen3_vl import PathStreamModel
from train.config import read_json, write_json
from train.optimization import backward_sessions, optimizer_and_schedule
from train.validation import CE_GROUPS, EarlyStopping, averages, move_batch, validate


def loaders(config, processor, rank, world):
    root = Path(config["dataset_root"])
    directory = root / config["data_dir"]

    def dataset(split):
        return SessionDataset(directory, config["model_key"], split, config["feature_root"], config["path_map"])

    train = dataset(config["train_split"])
    validation = dataset(config["validation_split"])
    attention = config["losses"]["diagnosis_attention"]
    enabled = attention["enabled"] and attention["weight"] != 0
    annotations = None
    if enabled and config["matches_file"]:
        annotations = FrameMatches(root / config["contexts_file"], root / config["matches_file"])
    collate = SessionCollator(processor.tokenizer, config["attention"], enabled, annotations)
    sampler = ShuffledEpochSampler(train, rank, world, config["seed"], config["accumulation"])
    options = dict(batch_size=1, num_workers=config["workers"], pin_memory=True, collate_fn=collate)
    if config["workers"]:
        options.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train, sampler=sampler, **options)
    val_loader = DataLoader(validation, sampler=validation_partition(validation, world)[rank],
                            generator=torch.Generator().manual_seed(config["seed"]), **options)
    return train_loader, val_loader


def train_step(ddp, batches, batch_size, full_batch, device, parameters):
    counts = torch.zeros(3, device=device)
    metrics = torch.zeros(6, device=device)
    attention = torch.zeros(4, device=device)

    def session_loss(network, batch):
        counts.add_(batch.pop("batch_counts").to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, detail, attention_detail = network(**move_batch(batch, device))
        metrics.add_(detail)
        attention.add_(attention_detail)
        return loss

    loss = backward_sessions(ddp, parameters, batches, session_loss, batch_size, full_batch)
    for value in (counts, metrics, attention):
        dist.all_reduce(value)
    return {"loss": loss, **averages(CE_GROUPS, metrics[:3], counts),
            **averages(ATTENTION_GROUPS, attention[:2], attention[2:])}


def train_stage(config, output):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    processor = make_processor(config["resume"] or config["model"], config["max_pixels"])
    training, validation = loaders(config, processor, rank, world)
    model = PathStreamModel(SimpleNamespace(**config), device)
    model.train()
    ddp = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                  gradient_as_bucket_view=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    full_batch = world * config["accumulation"]
    session_count = len(training.dataset)
    steps_per_epoch = math.ceil(session_count / full_batch)
    total_steps = steps_per_epoch * config["epochs"]
    optimizer, scheduler = optimizer_and_schedule(parameters, config, total_steps)
    selector = EarlyStopping()
    tracking = None
    if rank == 0 and config["wandb_project"]:
        import wandb
        tracking = wandb.init(project=config["wandb_project"], name=f"{output.parent.name}/{output.name}",
                              dir=str(output), config=config)
        tracking.define_metric("step")
        tracking.define_metric("*", step_metric="step")

    def log(split, step, record):
        if rank == 0:
            with (output / f"{split}.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **record}) + "\n")
            print(f"{split} step={step}/{total_steps} loss={record['loss']:.5f}", flush=True)
            if tracking is not None:
                tracking.log({"step": step, **{f"{split}/{k}": v for k, v in record.items() if v is not None}})

    step = 0
    for epoch in range(config["epochs"]):
        training.sampler.seed = config["seed"] + epoch
        iterator, remaining = iter(training), len(training)
        for update in range(steps_per_epoch):
            batch_size = min(full_batch, session_count - update * full_batch)
            batches = [next(iterator) for _ in range(min(config["accumulation"], remaining))]
            remaining -= len(batches)
            optimizer.zero_grad(set_to_none=True)
            record = train_step(ddp, batches, batch_size, full_batch, device, parameters)
            record.update(epoch=epoch + 1, grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters, 1.0)),
                          learning_rate=scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
            step += 1
            log("train", step, record)
            if step % config["eval_steps"] == 0 or update + 1 == steps_per_epoch:
                result = validate(model, validation, device)
                improved = selector.observe(result["loss"], step)
                if rank == 0 and improved:
                    destination = output / f"checkpoint-{step}"
                    model.save(destination, processor)
                    write_json(destination / "run_config.json", config)
                    write_json(output / "best_checkpoint.json", {
                        "checkpoint": str(destination), "step": step, "validation_loss": selector.best_loss})
                dist.barrier()
                log("validation", step, {**result, "best_step": selector.best_step,
                                         "bad_evaluations": selector.bad_evaluations})
                if selector.bad_evaluations >= config["patience"]:
                    break
        if selector.bad_evaluations >= config["patience"]:
            break
    if rank == 0:
        best = read_json(output / "best_checkpoint.json")
        write_json(output / "run_summary.json", {"best_checkpoint": best["checkpoint"],
                   "best_step": best["step"], "best_validation_loss": best["validation_loss"],
                   "steps": step, "early_stopped": selector.bad_evaluations >= config["patience"]})
        if tracking is not None:
            tracking.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        train_stage(read_json(args.config), args.config.parent)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
