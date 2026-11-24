"""Merge source-frame features and select each dataset's requested frame order."""
import json

import torch


def sequence_dim(name):
    return 1 if name == "deepstack_features" else 0


def feature_names(payload):
    return ["features", "deepstack_features"]


def select_frames(payload, indices):
    if indices == payload["source_frame_record_indices"]:
        return payload
    positions = {index: position for position, index in enumerate(payload["source_frame_record_indices"])}
    order = [positions[index] for index in indices]
    counts = (payload["image_grid_thw"].prod(-1) // 4).tolist()
    selected = dict(payload)
    for name in feature_names(payload):
        chunks = payload[name].split(counts, dim=sequence_dim(name))
        selected[name] = torch.cat([chunks[position] for position in order], dim=sequence_dim(name))
    selected["image_grid_thw"] = payload["image_grid_thw"][order]
    selected["source_frame_record_indices"] = indices
    return selected


def load_frame_payload(path):
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def load_session_features(path, manifest_path):
    payload = load_frame_payload(path)
    manifest = json.loads(manifest_path.read_text())
    indices = [frame["source_frame_record_index"] for frame in manifest["frames"]]
    return select_frames(payload, indices)
