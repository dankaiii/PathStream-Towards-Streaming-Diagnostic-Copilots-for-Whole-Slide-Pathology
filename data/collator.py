"""Prepare token labels and causal diagnosis evidence for the retained session window."""
import json

import torch

from data.conversation import (
    DEFAULT_MAX_CONTEXT_TOKENS, IMAGE, compile_session, encode_session, make_labels, training_window,
)
from data.geometry import token_weights


class FrameMatches:
    """Read per-frame match annotations keyed by session, frame, description and report."""
    def __init__(self, contexts_file, matches_file):
        with contexts_file.open() as stream:
            self.contexts = {row["id"]: (row["source_frame_record_indices"], row["spatial"]["frames"])
                             for row in map(json.loads, stream)}
        with matches_file.open() as stream:
            self.matches = {
                (row["source_session_id"], int(row["source_frame_record_index"]), row["caption"], row["report_path"]):
                row["matched"] for row in map(json.loads, stream)
            }

    def for_session(self, row, first_frame=0):
        context = self.contexts.get(row["id"])
        if context is None:
            return [], None
        indices, geometry = context
        meta = row["metadata"]
        matched = []
        turns = row["supervision"]["turns"][first_frame:]
        for i, turn in enumerate(turns, start=first_frame):
            if turn["has_description"]:
                description = row["messages"][turn["assistant_message_index"]]["content"]
                key = (meta["source_session_id"], int(indices[i]), description, meta["report_path"])
                label = self.matches.get(key)
                if label is None:
                    return [], geometry
                if label:
                    matched.append(i)
        return matched, geometry


def modality_target(kind, queries, weights, matched, cutoff, alpha):
    # Each matched frame contributes equal mass within its modality.
    domain = weights[0].new_zeros(weights[0].shape, dtype=torch.bool)
    for weight in weights:
        domain |= weight > 0
    keys = domain[:cutoff].nonzero().flatten()
    if not keys.numel():
        return None
    components = []
    for index in matched:
        component = weights[index][keys]
        mass = component.sum()
        if mass > 0:
            components.append(component / mass)
    if not components:
        return None
    target = (1 - alpha) * torch.stack(components).mean(0) + alpha / keys.numel()
    return {"kind": kind, "queries": queries, "keys": keys, "target": target}


class SessionCollator:
    def __init__(self, tokenizer, settings, enabled, annotations=None,
                 max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS):
        self.tokenizer = tokenizer
        self.settings = settings
        self.enabled = enabled
        self.annotations = annotations
        self.max_context_tokens = max_context_tokens

    def attention_targets(self, row, ids, spans, labels, kinds, grid, counts, first_frame=0):
        final = row["supervision"]["final_qa"]
        if not self.enabled or final is None:
            return []
        if "embedding_path" in row or "matched_frames" in row or self.annotations is None:
            # Session-local, zero-based frame indices. An omitted or empty list skips KL.
            matched = row.get("matched_frames", [])
            geometry = row.get("spatial", {}).get("frames")
        else:
            matched, geometry = self.annotations.for_session(row, first_frame)
        matched = [i - first_frame for i in matched if i >= first_frame]
        if not matched:
            return []
        cutoff, begin, end, _ = spans[final["assistant_message_index"]]
        queries = torch.arange(begin - 1, end - 1)
        queries = queries[(labels[queries] != -100) & (kinds[queries] == 3)]
        if not queries.numel():
            return []
        limit = self.settings["max_query_tokens"]
        if queries.numel() > limit:
            queries = queries[torch.linspace(0, queries.numel() - 1, limit).round().long()]

        image_id = self.tokenizer.convert_tokens_to_ids(IMAGE)
        positions = (ids == image_id).nonzero().flatten().split(counts)
        visual, text = [], []
        for frame, turn in enumerate(row["supervision"]["turns"][first_frame:]):
            pixels = torch.zeros(ids.numel())
            if geometry is None:
                pixels[positions[frame]] = 1
            else:
                _, h, w = grid[frame].tolist()
                pixels[positions[frame]] = torch.tensor(token_weights(geometry[first_frame + frame], h // 2, w // 2))
            visual.append(pixels)
            words = torch.zeros(ids.numel())
            if turn["has_description"]:
                _, start, stop, _ = spans[turn["assistant_message_index"]]
                words[start:stop] = 1
            text.append(words)
        items = []
        for kind, weights in enumerate((visual, text)):
            item = modality_target(kind, queries, weights, matched, cutoff, self.settings["background_alpha"])
            if item is not None:
                items.append(item)
        return items

    def __call__(self, batch):
        (raw, visual), = batch
        row = compile_session(raw)
        grid = visual["image_grid_thw"]
        counts = (grid.prod(-1) // 4).tolist()
        ids, spans = encode_session(row, self.tokenizer, counts)
        ids, spans, first_frame = training_window(row, ids, spans, self.max_context_tokens)
        visual_start = sum(counts[:first_frame])
        counts, grid = counts[first_frame:], grid[first_frame:]
        labels, kinds = make_labels(row, self.tokenizer, ids, spans, first_frame)
        ids = torch.tensor(ids)
        items = self.attention_targets(row, ids, spans, labels, kinds, grid, counts, first_frame)
        return {
            "input_ids": ids.unsqueeze(0), "attention_mask": torch.ones_like(ids).unsqueeze(0),
            "labels": labels.unsqueeze(0), "loss_kind": kinds.unsqueeze(0),
            "image_grid_thw": grid, "visual_features": visual["features"][visual_start:],
            "deepstack_features": visual["deepstack_features"][:, visual_start:], "attention_items": items,
            "batch_counts": torch.stack([((kinds == group) & (labels != -100)).sum() for group in (1, 2, 3)]),
        }
