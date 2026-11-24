"""Train LiveCC or MM-Duet2 with the PathStream Stage 1 optimization schedule."""
import argparse
from datetime import datetime, timedelta
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys

import yaml

from train.config import read_json, write_json
from train.tmux import launch_tmux


ROOT = Path(__file__).resolve().parents[1]


def train(run):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from data.domain_sft import read_frames, training_batch
    from model import livecc, mmduet2
    from model.domain_sft import load_model, save_adapter, set_training_mode
    from train.optimization import backward_sessions, optimizer_and_schedule
    from train.validation import EarlyStopping

    config = read_json(run / "run_config.json")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config["seed"])
    protocol = {"livecc": livecc, "mmduet2": mmduet2}[config["model"]]
    model, processor = load_model(protocol, config["model_path"], config)
    ddp = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                  gradient_as_bucket_view=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    data_root = Path(config["data_root"])
    with (data_root / config["train_file"]).open() as stream:
        rows = map(json.loads, stream)
        if config["session_id"]:
            rows = islice((row for row in rows if row["id"] == config["session_id"]), 1)
        training = list(islice(rows, config["steps"]))
    with (data_root / config["validation_file"]).open() as stream:
        validation = list(map(json.loads, islice(stream, rank, None, world)))

    session_budget = len(training) * config["epochs"]
    if config["steps"] is not None:
        session_budget = min(session_budget, config["steps"])
    if session_budget <= 0:
        raise ValueError("No training sessions selected")
    epoch_sizes = [min(len(training), session_budget - start)
                   for start in range(0, session_budget, len(training))]
    full_size = world * config["accumulation"]
    total_steps = sum(math.ceil(size / full_size) for size in epoch_sizes)
    optimizer, scheduler = optimizer_and_schedule(parameters, config, total_steps)
    selector = EarlyStopping()
    loss_options = {"skip_logits": True} if config["model"] == "livecc" else {}

    def session_loss(network, row):
        frames = read_frames(row, data_root, config["frame_reader_python"], config["path_map"])
        batch, _ = training_batch(protocol, processor, row, frames, config["max_pixels"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return network(**batch.to(device), use_cache=False, **loss_options).loss

    def log(split, record):
        if rank == 0:
            with (run / f"{split}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"{split} " + json.dumps(record), flush=True)

    def evaluate(step):
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.random.fork_rng(devices=[device]), torch.inference_mode():
            for row in validation:
                totals[0] += session_loss(model, row).detach()
        totals[1] = len(validation)
        dist.all_reduce(totals)
        loss_sum, sessions = totals.tolist()
        if not sessions:
            raise ValueError("The validation split contains no sessions")
        val_loss = loss_sum / sessions
        improved = selector.observe(val_loss, step)
        if rank == 0 and improved:
            destination = run / f"checkpoint-{step}"
            save_adapter(model, destination)
            processor.save_pretrained(destination)
            write_json(destination / "run_config.json", config)
            write_json(run / "best_checkpoint.json", {"checkpoint": str(destination),
                       "step": step, "validation_loss": val_loss})
        dist.barrier()
        log("validation", {"step": step, "loss": val_loss, "sessions": int(sessions),
                           "best_step": selector.best_step, "bad_evaluations": selector.bad_evaluations})
        set_training_mode(model)

    step = 0
    for epoch, size in enumerate(epoch_sizes):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(training), generator=generator).tolist()[:size]
        for start in range(0, size, full_size):
            indices = order[start:start + full_size]
            local_rows = [training[i] for i in indices[rank::world]]
            optimizer.zero_grad(set_to_none=True)
            loss = backward_sessions(ddp, parameters, local_rows, session_loss, len(indices), full_size)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            step += 1
            log("train", {"epoch": epoch + 1, "step": step, "sessions": len(indices),
                          "loss": loss, "learning_rate": lr, "grad_norm": grad_norm})
            if step % config["eval_steps"] == 0 or start + len(indices) == size:
                evaluate(step)
                if selector.bad_evaluations >= config["patience"]:
                    break
        if selector.bad_evaluations >= config["patience"]:
            break
    if rank == 0:
        best = read_json(run / "best_checkpoint.json")
        (run / "adapter").symlink_to(Path(best["checkpoint"]).name, target_is_directory=True)
        write_json(run / "run_summary.json", {"best_checkpoint": best["checkpoint"], "best_step": best["step"],
                   "best_validation_loss": best["validation_loss"], "steps": step,
                   "early_stopped": selector.bad_evaluations >= config["patience"]})
        print(f"Selected {best['checkpoint']} (validation loss {best['validation_loss']:.5f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("livecc", "mmduet2"))
    parser.add_argument("--model-path", type=Path, required=True, help="Local baseline checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing the SFT split")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for runs and adapters")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs (default from the training config)")
    parser.add_argument("--validation-split", help="Validation filename stem, for example val or dev")
    parser.add_argument("--frame-reader-python", help="Interpreter with Pillow and OpenSlide")
    parser.add_argument("--path-map", nargs=2, action="append", metavar=("OLD", "NEW"))
    parser.add_argument("--session-id", help="Train on the first occurrence of this session")
    parser.add_argument("--steps", type=int, help="Limit the total number of training sessions processed across epochs")
    parser.add_argument("--ssh-host", help="SSH alias for the progress command")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_dir:
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        try:
            train(args.worker_dir)
        finally:
            dist.destroy_process_group()
        return
    recipe = yaml.safe_load(args.config.read_text())
    stage = {**recipe, **recipe["stages"]["stage1"]}
    session = f"domain_sft_{args.model}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run = args.output_root.resolve() / session
    run.mkdir(parents=True)
    config = {"model": args.model, "model_path": str(args.model_path.resolve()),
              "data_root": str(args.data_root.resolve() / stage["data_dir"]),
              "train_file": stage["train_split"] + ".jsonl",
              "validation_file": (args.validation_split or stage["validation_split"]) + ".jsonl",
              "frame_reader_python": args.frame_reader_python or recipe["frame_reader_python"] or sys.executable,
              "path_map": args.path_map if args.path_map is not None else recipe["path_map"],
              "world_size": stage["world_size"] if args.num_gpus is None else args.num_gpus,
              "steps": args.steps, "session_id": args.session_id,
              **{key: stage[key] for key in ("learning_rate", "seed", "max_pixels", "lora_r", "lora_alpha",
                                             "epochs", "accumulation", "warmup_ratio", "eval_steps", "patience")}}
    write_json(run / "run_config.json", config)
    worker = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={config['world_size']}", "-m", "train.domain_sft",
              *sys.argv[1:], "--worker-dir", str(run)]
    launch_tmux(session, ROOT, run, worker, args.ssh_host)


if __name__ == "__main__":
    main()
