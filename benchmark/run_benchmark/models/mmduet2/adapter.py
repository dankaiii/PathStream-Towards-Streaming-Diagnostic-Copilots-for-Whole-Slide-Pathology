"""Base and domain-SFT checkpoints share the native mmduet2 inference protocol."""
from ..mmduet2.runtime import Backend, NAME, PROMPTS
from ...native import NativeAdapter
from ..choice import parse_answer


def load(model_path, source_dir, *, device="cuda:0", adapter_path=None, **settings):
    options = {**settings, "model_path": model_path}
    options['source_dir'] = source_dir
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": 256, "diagnosis_max_new_tokens": 256, "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    if adapter_path:
        from peft import PeftModel
        backend.model = PeftModel.from_pretrained(backend.model, adapter_path).merge_and_unload().eval()
    adapter = NativeAdapter(backend, PROMPTS)
    adapter.parse_choice = parse_answer
    return adapter
