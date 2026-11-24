"""Tokenize streaming conversations and align next-token supervision."""
import torch
from transformers import AutoProcessor

DEFAULT_MAX_CONTEXT_TOKENS = 24_576

IMAGE = "<|image_pad|>"
EOT = "<|im_end|>"
START = "<|im_start|>"


def make_processor(model_path, max_pixels):
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, min_pixels=3136, max_pixels=max_pixels,
    )
    # Fast processors load min/max overrides after construction; keep indexing
    # and saved processor metadata on the same pixel limits as image encoding.
    processor.image_processor.size = {"shortest_edge": 3136, "longest_edge": max_pixels}
    return processor


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def assistant_prefix(tokenizer):
    return encode(tokenizer, START + "assistant\n")


def frame_message():
    return {"role": "user", "content": "<image>"}


def compile_session(row):
    """Adapt VideoLLM stream blocks to complete Qwen image messages."""
    messages = [{"role": "system", "content": row["system"]}]
    turns, final_qa, cursor = [], None, 0
    for entry in row["conversation"]:
        if entry["role"] == "stream":
            for _ in range(entry["num_frames"]):
                turns.append({
                    "turn_index": cursor, "image_index": cursor, "time_sec": cursor / row["fps"],
                    "input_message_index": len(messages), "learn_stream": entry["learn"],
                    "has_description": row["supervision"]["frame_targets"][cursor] == "update",
                })
                messages.append(frame_message())
                cursor += 1
        elif entry["role"] == "user":
            if entry["kind"] == "description":
                turns[-1]["query_message_index"] = len(messages)
            else:
                final_qa = {"input_message_index": len(messages)}
            messages.append({"role": "user", "content": entry["content"]})
        elif entry["kind"] == "update":
            turns[-1]["assistant_message_index"] = len(messages)
            turns[-1]["learn_description"] = entry["learn"]
            messages.append({"role": "assistant", "content": entry["content"]})
        else:
            final_qa["assistant_message_index"] = len(messages)
            messages.append({"role": "assistant", "content": entry["content"]})
    return {**row, "messages": messages,
            "supervision": {**row["supervision"], "turns": turns, "final_qa": final_qa}}


def message_tokens(message, tokenizer, image_tokens=None):
    content = message["content"]
    if image_tokens is not None:
        content = content.replace(
            "<image>", "<|vision_start|>" + IMAGE * image_tokens + "<|vision_end|>", 1,
        )
    header = encode(tokenizer, START + message["role"] + "\n")
    body = encode(tokenizer, content)
    tokens = header + body + encode(tokenizer, EOT + "\n")
    return tokens, len(header), len(header) + len(body)


def encode_session(row, tokenizer, image_counts):
    image_messages = {
        turn["input_message_index"]: turn["image_index"]
        for turn in row["supervision"]["turns"]
    }
    ids, spans = [], {}
    for index, message in enumerate(row["messages"]):
        count = image_counts[image_messages[index]] if index in image_messages else None
        tokens, body_start, body_end = message_tokens(message, tokenizer, count)
        start = len(ids)
        ids.extend(tokens)
        spans[index] = (start, start + body_start, start + body_end, len(ids))
    return ids, spans


def turns_to_evict(excess, removable_sizes, budget):
    """Count oldest complete observations until the excess tokens fit."""
    for count, size in enumerate(removable_sizes, 1):
        excess -= size
        if excess <= 0:
            return count
    raise ValueError(f"Instructions and the newest complete turn exceed the {budget}-token context budget.")


def training_window(row, ids, spans, budget):
    """Keep text instructions, the most recent complete frames and the final QA."""
    excess = len(ids) - budget
    if excess <= 0:
        return ids, spans, 0

    turns = row["supervision"]["turns"]

    def observation_messages(turn):
        yield turn["input_message_index"]
        if turn["has_description"]:
            yield turn["assistant_message_index"]

    # The explicit morphology query is separate from its frame and reply.
    sizes = (sum(spans[i][3] - spans[i][0] for i in observation_messages(turn))
             for turn in turns[:-1])
    first_frame = turns_to_evict(excess, sizes, budget)
    removed = {i for turn in turns[:first_frame] for i in observation_messages(turn)}
    retained_ids, retained_spans = [], {}
    for i, span in spans.items():
        if i in removed:
            continue
        start, _, _, end = span
        shift = start - len(retained_ids)
        retained_spans[i] = tuple(position - shift for position in span)
        retained_ids.extend(ids[start:end])
    return retained_ids, retained_spans, first_frame


def make_labels(row, tokenizer, ids, spans, first_frame=0):
    # labels[i] is predicted from hidden[i]. The loss must not shift again.
    labels = torch.full((len(ids),), -100, dtype=torch.long)
    kinds = torch.zeros(len(ids), dtype=torch.long)
    eos_id = tokenizer.convert_tokens_to_ids(EOT)

    def supervise_answer(index, kind):
        start, _, body_end, _ = spans[index]
        # Learn the entire native assistant header, body, and reply EOS.
        labels[start - 1:body_end] = torch.tensor(ids[start:body_end + 1])
        kinds[start - 1:body_end] = kind

    for turn in row["supervision"]["turns"][first_frame:]:
        if turn["has_description"] and turn["learn_description"]:
            supervise_answer(turn["assistant_message_index"], 2)
        if not turn["learn_stream"]:
            # The first answer is supervised after its query, not after the frame.
            continue
        anchor = spans[turn["input_message_index"]][3] - 1
        if not turn["has_description"]:
            # This target is never inserted into input_ids/history.
            labels[anchor] = eos_id
        kinds[anchor] = 1
    final_qa = row["supervision"]["final_qa"]
    if final_qa is not None:
        supervise_answer(final_qa["assistant_message_index"], 3)
    return labels, kinds
