"""PathStream inference with the trained Qwen3-VL merger, DeepStack mergers and language LoRA."""
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoProcessor

from data.conversation import DEFAULT_MAX_CONTEXT_TOKENS, frame_message
from model.qwen3_vl import PathStreamModel, load_base_model
from model.inference import StreamingSession
from ...base import Adapter
from ...chat import SYSTEM


class InferenceModel(PathStreamModel):
    def __init__(self, model_path, checkpoint, device):
        torch.nn.Module.__init__(self)
        base, self.backend = load_base_model(model_path, device)
        self.adapter = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        weights = load_file(str(Path(checkpoint) / "merger_model.safetensors"))
        for name, merger in self.mergers().items():
            prefix = name + "."
            merger.float().load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        base.config.use_cache = True
        self.language_model.config.use_cache = True


def visual_frames(payload, indices):
    """Select source-frame record indices from a portable pre-merger feature payload."""
    from data.features.store import select_frames
    selected = select_frames(payload, list(indices))
    grids = selected["image_grid_thw"]
    counts = (grids.prod(-1) // 4).tolist()
    for features, grid, deepstack in zip(selected["features"].split(counts), grids,
                                         selected["deepstack_features"].split(counts, dim=1)):
        yield {"features": features, "grid": grid, "deepstack_features": deepstack}


class Session:
    def __init__(self, backend, system, instruction):
        self.device, self.instruction = backend.device, instruction
        self.stream = StreamingSession(backend.model, backend.tokenizer, backend.device,
                                       backend.eos_threshold, backend.max_context_tokens)
        self.stream.append_message({"role": "system", "content": system})

    def tensors(self, frame):
        return (frame["features"].to(self.device), frame["grid"].reshape(1, 3).to(self.device),
                frame["deepstack_features"].to(self.device))

    def append(self, frame):
        self.stream.append_message(frame_message(), *self.tensors(frame))

    def observe(self, frame, index, fps, budget):
        if index == 0:
            self.append(frame)
            result = self.diagnose(self.instruction, budget)
            result["decision"] = "protocol_error" if result["protocol_error"] else "update" if result["response"] else "silent"
            return result
        features, grid, deepstack = self.tensors(frame)
        return self.stream.observe(frame_message(), features, grid, budget, deepstack)

    def diagnose(self, question, budget):
        return self.stream.answer({"role": "user", "content": question}, budget)


class PathStream(Adapter):
    native = True

    def __init__(self, model_path, checkpoint, device, eos_threshold, max_context_tokens):
        self.device = torch.device(device)
        self.model = InferenceModel(model_path, checkpoint, device).eval().requires_grad_(False)
        self.tokenizer = AutoProcessor.from_pretrained(checkpoint).tokenizer
        self.eos_threshold, self.max_context_tokens = eos_threshold, max_context_tokens

    def start(self, task, system, instruction):
        return Session(self, system, instruction)

    def answer_frames(self, frames, prompt, budget):
        session = Session(self, SYSTEM, "")
        for frame in frames:
            session.append(frame)
        return session.diagnose(prompt, budget)


def load(model_path, checkpoint, device="cuda:0", eos_threshold=0.0, max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS):
    return PathStream(model_path, checkpoint, device, eos_threshold, max_context_tokens)
