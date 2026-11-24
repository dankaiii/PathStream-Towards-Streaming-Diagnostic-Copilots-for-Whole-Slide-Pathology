"""Tasks 1–4 for qwen3.8-flash. Supply a configured API client."""
from ..hosted import Hosted


def load(client, model="qwen3.8-flash"):
    return Hosted(client, model, responses=False)
