"""Tasks 1–4 for Qwen3-VL-8B-Instruct."""
from ..local import Local


def load(model_path, device="auto", adapter_path=None):
    model = Local(model_path, "Qwen3VLForConditionalGeneration", device, adapter_path)
    return model
