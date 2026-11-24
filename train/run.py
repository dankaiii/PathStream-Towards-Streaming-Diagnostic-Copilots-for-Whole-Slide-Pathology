"""Launch two-stage training or one of the Stage 1 ablations in tmux."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml

from train.config import add_data_arguments, load_config, read_json, write_json
from train.tmux import launch_tmux


ROOT = Path(__file__).resolve().parents[1]


def launch(args, presets):
    config = load_config(args)
    stage, ablation = args.stage, args.ablation
    stages = config.pop("stages")
    names = list(presets) if ablation == "all" else [ablation or "full"]
    session = "pathstream_" + (ablation or stage) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = Path(config["output_root"]).resolve() / session
    run.mkdir(parents=True)
    if ablation:
        stage = "stage1"
    stage_names = ("stage1", "stage2") if stage == "both" else (stage,)
    jobs = []
    for name in names:
        paths = []
        for stage_name in stage_names:
            recipe = {**config, **stages[stage_name], "stage": stage_name,
                      "resume": str(args.init_from.resolve()) if args.init_from else None}
            changes = presets[name] if ablation else {}
            for key, value in changes.items():
                if key == "losses":
                    recipe["losses"] = {loss: {**options, **value.get(loss, {})}
                                        for loss, options in recipe["losses"].items()}
                else:
                    recipe[key] = value
            output = run / name / stage_name
            output.mkdir(parents=True)
            path = output / "run_config.json"
            write_json(path, recipe)
            paths.append(str(path))
        jobs.append(paths)
    write_json(run / "stages.json", jobs)
    worker = [sys.executable, "-u", "-m", "train.run", "--execute", str(run)]
    launch_tmux(session, ROOT, run, worker, args.ssh_host)


def execute(run):
    for job in read_json(run / "stages.json"):
        selected = None
        for filename in job:
            path = Path(filename)
            recipe = read_json(path)
            if selected is not None:
                recipe["resume"] = selected
                write_json(path, recipe)
            print(f"Training {path.parent.relative_to(run)}", flush=True)
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                            f"--nproc_per_node={recipe['world_size']}", "-m", "train.trainer",
                            "--config", str(path)], cwd=ROOT, check=True)
            selected = read_json(path.parent / "run_summary.json")["best_checkpoint"]
            print(f"Selected {selected}", flush=True)
    print("Training complete", flush=True)


def main():
    presets = yaml.safe_load((ROOT / "configs/ablations.yaml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--output-root", help="Parent directory for runs, checkpoints and logs")
    parser.add_argument("--num-gpus", dest="world_size", type=int, help="Number of GPUs per run")
    parser.add_argument("--contexts-file", help="Optional frame contexts JSONL, relative to the data root")
    parser.add_argument("--matches-file", help="Optional frame matches JSONL, relative to the data root")
    parser.add_argument("--ssh-host", help="SSH alias to use in the printed progress command")
    parser.add_argument("--stage", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument("--init-from", type=Path, help="Stage 1 checkpoint for standalone Stage 2 training")
    parser.add_argument("--ablation", choices=[*presets, "all"])
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    else:
        if args.ablation and args.stage == "stage2":
            parser.error("Ablations train Stage 1.")
        if (args.stage == "stage2") != bool(args.init_from):
            parser.error("--stage stage2 requires --init-from CHECKPOINT; other stages start from the base model.")
        launch(args, presets)


if __name__ == "__main__":
    main()
