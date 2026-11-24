"""Training configuration and run records."""
import json
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def add_data_arguments(parser):
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/train.yaml")
    parser.add_argument("--data-root", dest="dataset_root", help="Directory containing the shared training and validation files")
    parser.add_argument("--model-path", dest="model", help="Local base model and processor directory")
    parser.add_argument("--feature-root", help="Base directory for cached embedding paths")
    parser.add_argument("--validation-split", help="Validation filename stem for both stages")
    parser.add_argument("--path-map", nargs=2, action="append", metavar=("OLD", "NEW"),
                        help="Relocate an absolute path prefix stored in data records (repeatable)")


def load_config(args):
    config = yaml.safe_load(args.config.read_text())
    for key in ("dataset_root", "model", "feature_root", "output_root", "world_size", "contexts_file",
                "matches_file", "frame_reader_python", "path_map"):
        override = getattr(args, key, None)
        if override is not None:
            config[key] = override
    for key in ("dataset_root", "model", "feature_root", "output_root"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config["path_map"] = [[old, str(Path(new).expanduser().resolve())]
                          for old, new in config.get("path_map", [])]
    if args.validation_split:
        for stage in config["stages"].values():
            stage["validation_split"] = args.validation_split
    return config
