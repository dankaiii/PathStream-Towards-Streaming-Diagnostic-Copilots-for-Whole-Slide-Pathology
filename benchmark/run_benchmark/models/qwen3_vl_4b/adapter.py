"""Tasks 1–4 for Qwen3-VL-4B-Instruct."""
from ..local import Local


def load(model_path, device="auto"):
    return Local(model_path, "Qwen3VLForConditionalGeneration", device)
