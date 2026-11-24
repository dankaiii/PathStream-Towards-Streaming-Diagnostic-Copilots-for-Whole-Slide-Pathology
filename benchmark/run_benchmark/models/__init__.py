"""Load only the selected adapter and its model-specific dependencies."""
from importlib import import_module
from ..streaming_environments import MODEL_ENV


def load(name, **options):
    if name not in MODEL_ENV:
        raise ValueError(f"Unknown benchmark model: {name}")
    return import_module(f"{__name__}.{name}.adapter").load(**options)
