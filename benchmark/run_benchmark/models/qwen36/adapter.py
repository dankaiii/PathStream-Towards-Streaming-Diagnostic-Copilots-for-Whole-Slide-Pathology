"""Tasks 1–4 for qwen3.6-35b-a3b. Supply a configured API client."""
from ..hosted import Hosted


def load(client, model="qwen3.6-35b-a3b"):
    return Hosted(client, model, responses=False)
