"""Task 3: ask the final diagnostic question in the same live session."""
from .runner import stream


def run(model, sample, frames):
    questions = {turn["kind"]: turn["content"] for turn in sample["conversation"]
                 if turn["role"] == "user"}
    return stream(model, 3, frames, sample["fps"], sample["system"],
                  questions["description"], questions["diagnosis"])
