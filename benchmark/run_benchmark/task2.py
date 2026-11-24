"""Task 2: morphology descriptions over one clip with fresh history."""
from .runner import stream


def run(model, sample, frames, fps=1):
    return stream(model, 2, frames, sample.get("fps", fps), sample["system"], sample["instruction"])
