"""Base and domain-SFT checkpoints share the native livecc inference protocol."""
from ..livecc.runtime import Backend, NAME, PROMPTS
from ...native import NativeAdapter


DEFAULTS = {
    "attention": 'flash_attention_2',
    "max_pixels": 301056,
    "repetition_penalty": 1.05,
}


def load(model_path, *, device="cuda:0", adapter_path=None, **settings):
    options = {**DEFAULTS, **settings, "model_path": model_path}
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": 256, "diagnosis_max_new_tokens": 256, "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    if adapter_path:
        from peft import PeftModel
        backend.model = PeftModel.from_pretrained(backend.model, adapter_path).merge_and_unload().eval()
    adapter = NativeAdapter(backend, PROMPTS)
    return adapter
