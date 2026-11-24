"""Load indexed, complete sessions and their cached visual features."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from data.features.paths import resolve_reference
from data.features.store import load_frame_payload, load_session_features, select_frames


def split_entries(directory, model_key, split):
    with (directory / f"{split}.jsonl").open() as stream:
        rows = list(map(json.loads, stream))
    if rows and "embedding_path" in rows[0]:
        # The sampler only needs relative workload to distribute cached sessions.
        return rows, [{"row_index": i, "tokens": len(row["embedding_frame_indices"])}
                      for i, row in enumerate(rows)]
    index = directory / "data_index" / model_key / f"{split}.index.jsonl"
    with index.open() as stream:
        entries = list(map(json.loads, stream))
    return rows, entries


class SessionDataset(Dataset):
    def __init__(self, directory, model_key, split, feature_root, path_map=()):
        self.directory = directory
        self.path_map = path_map
        self.rows, entries = split_entries(directory, model_key, split)
        self.feature_root = Path(feature_root)
        exclusions = self.feature_root / "excluded_sessions.json"
        excluded = set(json.loads(exclusions.read_text())["session_ids"]) if exclusions.exists() else set()
        self.entries = [entry for entry in entries if self.rows[entry["row_index"]].get("id") not in excluded]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.rows[self.entries[index]["row_index"]]
        if "embedding_path" in row:
            path = resolve_reference(row["embedding_path"], self.feature_root, self.path_map)
            payload = load_frame_payload(path)
            return row, select_frames(payload, row["embedding_frame_indices"])
        manifest = resolve_reference(row["metadata"]["frame_manifest"], self.directory, self.path_map)
        features = load_session_features(self.feature_root / (row["id"] + ".pt"), manifest)
        return row, features
