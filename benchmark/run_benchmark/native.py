"""Native streaming sessions consume one projected observation at a time.

VideoLLM-online uses a [10, hidden] frame tensor. Other native baselines take
{"features": projected_tokens, "grid": thw}. SFT features must use that checkpoint's
visual projector. PathStream has its own pre-merger input adapter.
"""
from .base import Adapter


class NativeAdapter(Adapter):
    native = True

    def __init__(self, backend, prompts):
        self.backend, self.prompts = backend, prompts
        self.description_budget = backend.config["description_max_new_tokens"]
        self.diagnosis_budget = backend.config["diagnosis_max_new_tokens"]
        self.choice_budget = backend.config["answer_max_new_tokens"]

    def start(self, task, system, instruction):
        self.backend.instruction = self.prompts[task]
        return self.backend.session(system)

    def answer_frames(self, frames, prompt, budget):
        return self.backend.answer_frames(frames, prompt, budget)

    def end_case(self):
        self.backend.end_case()

    def close(self):
        self.backend.close()


class Runtime:
    def end_case(self):
        pass

    def close(self):
        self.end_case()
