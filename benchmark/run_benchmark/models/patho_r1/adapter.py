"""Tasks 1–4 for Patho-R1-7B."""
from ..local import Local
from ..choice import MAX_NEW_TOKENS, parse_answer


def load(model_path, device="auto", adapter_path=None):
    model = Local(model_path, "Qwen2_5_VLForConditionalGeneration", device, adapter_path)
    model.choice_budget = MAX_NEW_TOKENS
    model.parse_choice = parse_answer
    return model
