"""Tasks 1–4 for InternVL3_5-8B."""
from ..local import Local


def load(model_path, device="auto", adapter_path=None):
    model = Local(model_path, "InternVLChatModel", device, adapter_path)
    return model
