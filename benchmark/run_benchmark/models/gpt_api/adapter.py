"""Tasks 1–4 for gpt-6-luna. Supply a configured API client."""
from ..hosted import Hosted


def load(client, model="gpt-6-luna"):
    return Hosted(client, model, responses=True)
