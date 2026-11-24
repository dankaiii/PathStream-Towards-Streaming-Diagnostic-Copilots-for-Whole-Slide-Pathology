"""Tasks 1–4 for Lingshu-32B."""
from ..local import Local


def load(model_path, device="auto", adapter_path=None):
    model = Local(model_path, "Qwen2_5_VLForConditionalGeneration", device, adapter_path)
    return model
