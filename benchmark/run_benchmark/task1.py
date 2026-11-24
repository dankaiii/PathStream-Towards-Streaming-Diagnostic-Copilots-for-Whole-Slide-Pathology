"""Task 1: response timing over one clip with fresh history."""
from .runner import stream


def run(model, sample, frames, fps=1):
    return stream(model, 1, frames, sample.get("fps", fps), sample["system"], sample["instruction"])
