"""Resolve file references relative to their owner or an explicit relocation map."""
from pathlib import Path


def resolve_reference(value, directory, path_map):
    path = Path(value)
    if path.is_absolute():
        for old, new in path_map:
            if path.is_relative_to(old):
                return Path(new) / path.relative_to(old)
    return directory / path
