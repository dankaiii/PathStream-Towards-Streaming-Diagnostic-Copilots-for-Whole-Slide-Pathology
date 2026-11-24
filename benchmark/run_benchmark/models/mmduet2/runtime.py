"""MMDuet2. Keep native NO REPLY turns and the original Qwen2.5-VL KV state."""
from functools import wraps
from pathlib import Path
import sys
from ...native import Runtime


NAME = "mmduet2"


SYSTEM = (
    "You are a helpful assistant observing continuously incoming pathology frames. "
    "Use NO REPLY when the current view does not call for another description."
)


PROMPTS = {
    1: (
        'Observe the pathology views in their presented order. Briefly describe the morphology of '
        'each newly presented view in English. Describe the first frame. Output NO REPLY while a '
        'similar view continues after your description. A finding may be described again in '
        'another view; do not restrict descriptions to facts absent from the entire history. Do '
        'not give a diagnosis.'
    ),
    2: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Output NO '
        'REPLY while a similar view continues after your description. A finding may be described '
        'again in another view; do not restrict descriptions to facts absent from the entire '
        'history. Do not give a diagnosis.'
    ),
    3: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Output NO '
        'REPLY while a similar view continues after your description. A finding may be described '
        'again in another view; do not restrict descriptions to facts absent from the entire '
        'history. Give a diagnosis only when explicitly asked at the end.'
    ),
}


WAIT_TEXT = "NO REPLY"


class MMDuet2(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None):
        import torch
        from transformers import AutoProcessor
        sys.path.insert(0, str(Path(config[NAME]['source_dir']) / 'proactive_eval'))
        from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        self.torch, self.config, self.settings = torch, config, config[NAME]
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.system, self.instruction = system, instruction
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.settings["model_path"], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map=self.settings.get("device_map", {"": device}), attn_implementation="flash_attention_2",
        ).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(self.settings["model_path"], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.context_limit = self.model.config.max_position_embeddings
        self.active_session = None

    def close_session(self):
        self.active_session = None
        self.model.reset_status()


class NativeSession:
    def __init__(self, backend, system):
        self.backend, self.model, self.processor = backend, backend.model, backend.processor
        self.tokenizer, self.torch, self.device = backend.tokenizer, backend.torch, backend.device
        self.history = [{"role": "system", "content": system}]
        self.images, self.past = [], None
        self.token_count = 0
        self.image_tokens_seen = 0

    def frame_inputs(self, image, index):
        content = [{"type": "image", "image": image}]
        if index == 0:
            content.append({"type": "text", "text": self.backend.instruction})
        return self.prepare(content)

    def generate(self, inputs, max_new_tokens, diagnosis=False):
        input_length = inputs.input_ids.shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, past_key_values=self.past, use_cache=True, return_dict_in_generate=True,
                max_new_tokens=max_new_tokens, do_sample=False,
                drop_method="none", drop_threshold=1.0, drop_absolute=True,
            )
        self.past = output.past_key_values
        tokens = output.sequences[0, input_length:]
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        body = decoded.strip()
        # Keep native waiting turns in history, just as the official inference client does.
        self.history.append({"role": "assistant", "content": decoded})
        self.token_count = output.sequences.shape[1]
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        stopped = tokens[-1].item() in eos
        waiting = body == WAIT_TEXT
        # A native NO REPLY is an empty prediction, not an execution failure.
        bad = not stopped or not body
        total_visual = int((inputs.input_ids == self.model.config.image_token_id).sum().item())
        visual_tokens = total_visual - self.image_tokens_seen
        self.image_tokens_seen = total_visual
        self.torch.cuda.synchronize(self.device)
        result = {
            "response": "" if waiting else body,
            "raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "decision": "protocol_error" if bad else "answer" if diagnosis else "silent" if waiting else "update",
            "protocol_error": bad, "input_tokens": input_length, "generated_tokens": len(tokens),
            "context_tokens_after_turn": self.token_count, "visual_tokens": visual_tokens,
            "native_wait_marker": waiting, "stopped_on_eos": stopped,
            "hit_generation_limit": not stopped, "stop_reason": "eos" if stopped else "generation_limit",
        }
        return result

    def frame(self, image, index, fps, max_new_tokens):
        return self.generate(self.frame_inputs(image, index), max_new_tokens)

    def diagnose(self, question, max_new_tokens):
        return self.generate(self.prepare([{"type": "text", "text": question}]), max_new_tokens, diagnosis=True)


class Session(NativeSession):
    def frames(self):
        return [item['image'] for msg in self.history if isinstance(msg['content'], list)
                for item in msg['content'] if item['type'] == 'image']

    def cached_inputs(self):
        from transformers.feature_extraction_utils import BatchFeature
        frames = self.frames()
        text = self.processor.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        parts = text.split(self.processor.image_token)
        expanded = parts[0]
        for frame, tail in zip(frames, parts[1:]):
            expanded += self.processor.image_token * len(frame['features']) + tail
        tokens = self.tokenizer([expanded], return_tensors='pt')
        return BatchFeature({**tokens,
            'pixel_values': self.torch.cat([frame['features'] for frame in frames]),
            'image_grid_thw': self.torch.stack([frame['grid'] for frame in frames]),
        }).to(self.device)

    def prepare(self, content):
        self.history.append({'role': 'user', 'content': content})
        return self.cached_inputs()

    def observe(self, frame, index, fps, budget):
        return {**self.frame(frame, index, fps, budget),
                'visual_tokens': len(frame['features'])}


class Backend(MMDuet2):
    def __init__(self, config, device, instruction=None):
        super().__init__(config, device, instruction=instruction)
        # Native input preparation slices pixels by raw patch counts. Select the
        # projected features by the image tokens in its already-sliced input instead.
        prepare = self.model.prepare_inputs_for_generation
        @wraps(prepare)
        def cached_prepare(*args, **kwargs):
            features = kwargs.get('pixel_values')
            result = prepare(*args, **kwargs)
            if result['pixel_values'] is not None:
                count = int((result['input_ids'] == self.model.config.image_token_id).sum())
                result['pixel_values'] = features[-count:]
            return result
        self.model.prepare_inputs_for_generation = cached_prepare
        self.model.visual.forward = lambda hidden_states, grid_thw: hidden_states

    def session(self, tissue_context=''):
        self.close_session()
        self.active_session = Session(self, self.system + '\n' + tissue_context)
        return self.active_session

    def answer_frames(self, frames, prompt, budget):
        session = self.session()
        inputs = session.prepare([*({'type': 'image', 'image': frame} for frame in frames),
                                  {'type': 'text', 'text': prompt}])
        return session.generate(inputs, budget, diagnosis=True)

    def end_case(self):
        self.close_session()
