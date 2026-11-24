"""Read raw session frames and construct native assistant-token SFT targets."""
import json
from pathlib import Path
import subprocess

import torch
from PIL import Image

from data.conversation import DEFAULT_MAX_CONTEXT_TOKENS, compile_session
from data.features.paths import resolve_reference


def read_frames(row, data_root, reader_python, path_map):
    manifest = resolve_reference(row["metadata"]["frame_manifest"], data_root, path_map)
    document = json.loads(manifest.read_text())
    indices = range(len(row["supervision"]["frame_targets"]))
    reader = Path(__file__).parent / "features/read_frames.py"
    command = [reader_python, str(reader), str(manifest), "--frame-indices", *map(str, indices)]
    for old, new in path_map:
        command.extend(["--path-map", old, new])
    width, height = document["width"], document["height"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        frames = [Image.frombytes("RGB", (width, height), process.stdout.read(width * height * 3))
                  for _ in indices]
        process.stdout.close()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    return frames


def window_messages(protocol, session, first_frame):
    source = session["messages"]
    messages, kinds = [source[0]], []
    for turn in session["supervision"]["turns"]:
        instruction = None
        if "query_message_index" in turn:
            prompt = source[turn["query_message_index"]]["content"]
            instruction = prompt.replace("Remain silent", protocol.SILENCE_INSTRUCTION)
        if turn["image_index"] < first_frame:
            if instruction:
                messages.append({"role": "user", "content": instruction})
            continue
        content = protocol.frame_content(turn["image_index"], session["fps"])
        if instruction:
            content.append({"type": "text", "text": instruction})
        if turn["has_description"]:
            reply = source[turn["assistant_message_index"]]["content"] + protocol.DESCRIPTION_SUFFIX
            kind = "description"
        else:
            reply, kind = protocol.SILENCE, "silence"
        messages.extend([{"role": "user", "content": content}, {"role": "assistant", "content": reply}])
        kinds.append(kind)
    qa = session["supervision"]["final_qa"]
    if qa is not None:
        messages.extend([source[qa["input_message_index"]], source[qa["assistant_message_index"]]])
        kinds.append("diagnosis")
    return messages, kinds


def conversation(protocol, row):
    return window_messages(protocol, compile_session(row), 0)


def training_batch(protocol, processor, row, frames, max_pixels, max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS):
    session = compile_session(row)
    # Measure the native processor's expanded visual and text tokens. For a long
    # session, find the longest suffix of complete observations that fits.
    first_frame, low, high = 0, 0, len(frames) - 1
    while True:
        messages, kinds = window_messages(protocol, session, first_frame)
        batch = protocol.encode(processor, messages, frames[first_frame:], max_pixels)
        if batch.input_ids.shape[1] <= max_context_tokens:
            if first_frame == low:
                break
            high = first_frame
        else:
            if first_frame >= high:
                raise ValueError(f"Instructions and newest complete turn exceed {max_context_tokens} tokens.")
            low = first_frame + 1
        del batch
        first_frame = (low + high) // 2
    ids = batch.input_ids[0]
    tokenizer = processor.tokenizer
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    starts = (ids.unfold(0, len(header), 1) == torch.tensor(header)).all(-1).nonzero().flatten()
    ends = (ids == eos).nonzero().flatten()
    labels = torch.full_like(ids, -100)
    target_counts = []
    for kind, start in zip(kinds, starts.tolist(), strict=True):
        begin = start + len(header)
        end = int(ends[ends >= begin][0]) + 1
        # Both upstream objectives score assistant text and its EOS, including
        # the complete native silence reply. All non-assistant tokens stay masked.
        labels[begin:end] = ids[begin:end]
        target_counts.append({"kind": kind, "tokens": end - begin})
    batch["labels"] = labels.unsqueeze(0)
    return batch, target_counts
