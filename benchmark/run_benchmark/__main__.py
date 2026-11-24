"""Run one public example with your own ordered images or native visual features."""
import argparse
from importlib import import_module
import json
import os
from pathlib import Path

from .models import load
from .streaming_environments import MODEL_ENV


def read_images(paths):
    from PIL import Image
    for path in paths:
        with Image.open(path) as image:
            yield image.convert("RGB")


def read_features(name, path, indices):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    if name == "pathstream":
        from .models.pathstream.adapter import visual_frames
        yield from visual_frames(data, data["source_frame_record_indices"] if indices is None else indices)
    elif name == "videollm_online":
        frames = data["features"]
        for i in range(len(frames)) if indices is None else indices:
            yield frames[i]
    elif name in ("livecc", "livecc_sft"):
        grids, offsets = data["video_grid_thw"], data["frame_token_offsets"]
        for i in range(len(grids)) if indices is None else indices:
            yield {"grid": grids[i], "features": data["features"][offsets[i]:offsets[i + 1]]}
    else:
        frames = data["frames"]
        for i in range(len(frames)) if indices is None else indices:
            yield frames[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_ENV)
    parser.add_argument("task", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--sample", type=Path, required=True, help="One release-format JSON example")
    parser.add_argument("--model-options", type=Path, help="JSON keyword arguments for the selected adapter's load function")
    parser.add_argument("--fps", type=float, default=1, help="Presentation rate for Task 1/2 clips without an explicit fps (default: %(default)s)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, nargs="+", help="Ordered images for general and medical chat adapters")
    inputs.add_argument("--features", type=Path, help="This streaming model's native feature payload")
    parser.add_argument("--indices", type=int, nargs="+", help="Feature positions (source record indices for PathStream)")
    args = parser.parse_args()
    options = json.loads(args.model_options.read_text()) if args.model_options else {}
    if MODEL_ENV[args.model] == "pathstream-api":
        from openai import OpenAI
        provider = "OPENAI" if args.model == "gpt_api" else "DASHSCOPE"
        default_url = None if provider == "OPENAI" else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        endpoint = options.pop("base_url", os.environ.get(provider + "_BASE_URL", default_url))
        options["client"] = OpenAI(base_url=endpoint, api_key=os.environ[provider + "_API_KEY"])
    model = load(args.model, **options)
    frames = read_images(args.images) if args.images else read_features(args.model, args.features, args.indices)
    sample = json.loads(args.sample.read_text())
    task = import_module(f".task{args.task}", __package__)
    try:
        predictions = (task.run(model, sample, frames, args.fps) if args.task in (1, 2)
                       else task.run(model, sample, frames))
        print(json.dumps(predictions, ensure_ascii=False))
    finally:
        model.close()


if __name__ == "__main__":
    main()
