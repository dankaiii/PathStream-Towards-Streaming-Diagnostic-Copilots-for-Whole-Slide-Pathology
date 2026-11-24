"""LiveCC-7B-Instruct. Preserve native pause turns, KV cache and timestamped video inputs."""
import functools
from ...native import Runtime


NAME = "livecc"


SYSTEM = "You are a helpful assistant."


PROMPTS = {
    1: (
        'Observe the pathology views in their presented order. Briefly describe the morphology of '
        'each newly presented view in English. Describe the first frame. Pause while a similar '
        'view continues after your description. A finding may be described again in another view; '
        'do not restrict descriptions to facts absent from the entire history. Do not give a '
        'diagnosis.'
    ),
    2: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Pause '
        'while a similar view continues after your description. A finding may be described again '
        'in another view; do not restrict descriptions to facts absent from the entire history. '
        'Do not give a diagnosis.'
    ),
    3: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Pause '
        'while a similar view continues after your description. A finding may be described again '
        'in another view; do not restrict descriptions to facts absent from the entire history. '
        'Give a diagnosis only when explicitly asked at the end.'
    ),
}


class LiveCC(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None):
        import torch
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

        # The official inference code applies this patch before constructing Qwen2-VL.
        apply_liger_kernel_to_qwen2_vl()
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation

        self.torch, self.config = torch, config
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.settings = config[NAME]
        self.system, self.instruction = system, instruction
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.settings["model_path"], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": str(self.device)}, attn_implementation=self.settings["attention"],
        ).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            self.settings["model_path"], local_files_only=True, use_fast=False,
            max_pixels=self.settings["max_pixels"],
        )
        self.tokenizer = self.processor.tokenizer
        self.pause_id = self.tokenizer(" ...").input_ids[-1]
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else eos
        self.model.prepare_inputs_for_generation = functools.partial(
            prepare_multiturn_multimodal_inputs_for_generation, self.model,
        )
        self.model.config.use_cache = True
        self.context_limit = self.model.config.text_config.max_position_embeddings


class NativeSession:
    def __init__(self, backend, system):
        self.backend, self.model = backend, backend.model
        self.torch, self.device = backend.torch, backend.device
        self.processor, self.tokenizer = backend.processor, backend.tokenizer
        self.system = system
        self.past_ids = None
        self.past = None
        self.last_logits = None
        self.model.model.rope_deltas = None

    @property
    def token_count(self):
        return 0 if self.past_ids is None else self.past_ids.shape[1]

    def inputs(self, content, frames=None):
        import numpy as np

        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.past_ids is not None:
            # generate() leaves its final EOS outside the cache. Follow the official
            # demo: supply that ChatML boundary together with the next user message.
            text = "<|im_end|>\n" + text[text.index("<|im_start|>user"):]
        videos = None if frames is None else [np.stack([np.asarray(frame) for frame in frames])]
        inputs = self.processor(
            text=text, videos=videos, return_tensors="pt", return_attention_mask=False,
            # Transformers 4.57 has a separate video processor. Its resize uses size,
            # not the legacy max_pixels attribute; keep the official demo's budget.
            videos_kwargs={
                "do_sample_frames": False,
                "size": {"shortest_edge": self.processor.image_processor.min_pixels,
                         "longest_edge": self.backend.settings["max_pixels"]},
            },
        ).to(self.device)
        if self.past_ids is not None:
            inputs["input_ids"] = self.torch.cat([self.past_ids, inputs.input_ids], dim=1)
        return inputs

    def generate(self, inputs, max_new_tokens, diagnosis=False):
        input_tokens = inputs.input_ids.shape[1]
        # Count only this chunk's video tokens, not the retained token history.
        start = self.token_count
        visual_tokens = int((inputs.input_ids[:, start:] == self.model.config.video_token_id).sum())
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, past_key_values=self.past, use_cache=True,
                return_dict_in_generate=True, do_sample=False,
                repetition_penalty=self.backend.settings["repetition_penalty"],
                max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
            )
        self.past = output.past_key_values
        self.past_ids = output.sequences[:, :-1]
        self.last_generated_ids = output.sequences[:, input_tokens:].clone()
        tokens = output.sequences[0, input_tokens:].tolist()
        stopped = tokens[-1] in self.backend.eos_ids
        body = tokens[:-1] if stopped else tokens
        native_pause = bool(body and body[-1] == self.backend.pause_id)
        if native_pause:
            body = body[:-1]
        text = self.tokenizer.decode(body, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
        bad = not stopped or (diagnosis and not text)
        return {
            "response": text, "raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False),
            # Labels below belong to the benchmark result schema, never to the prompt.
            "decision": "protocol_error" if bad else "answer" if diagnosis else "update" if text else "silent",
            "protocol_error": bad, "input_tokens": input_tokens, "generated_tokens": len(tokens),
            "context_tokens_after_turn": output.sequences.shape[1],
            "stopped_on_eos": stopped, "hit_generation_limit": not stopped,
            "native_pause": native_pause,
            "stop_reason": "generation_limit" if not stopped else "native_pause_then_eos" if native_pause else "eos",
            "visual_tokens": visual_tokens,
        }


def window_metadata(session):
    return {'window_frames': sum(len(block['frames']) for block in session.blocks),
            'evicted_frames_this_turn': session.evicted_this_turn,
            'evicted_frames_total': session.evicted_frames,
            'max_context_tokens': session.backend.context_limit,
            'kv_cache_reset_on_window_eviction': bool(session.evicted_this_turn)}


class Session(NativeSession):
    def __init__(self, backend, system, observations=True):
        super().__init__(backend, system)
        self.blocks = []
        self.evicted_frames = self.evicted_this_turn = 0
        messages = [{'role': 'system', 'content': system}]
        if observations:
            messages.append({'role': 'user', 'content': backend.instruction})
        header = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        self.header_ids = self.tokenizer(header, return_tensors='pt', return_attention_mask=False).input_ids
        self.boundary_ids = self.tokenizer('<|im_end|>\n', return_tensors='pt',
                                           return_attention_mask=False).input_ids

    def remember_input(self, text, content, frames):
        text = text[text.index('<|im_start|>user'):]
        self.current_block = {'prompt_ids': self.tokenizer(text, return_tensors='pt',
                               return_attention_mask=False).input_ids,
                              'frames': list(frames), 'content': list(content)}

    def inputs(self, content, frames=None):
        inputs = super().inputs(content, frames)
        text = self.processor.apply_chat_template(
            [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True)
        self.remember_input(text, content, [])
        return inputs

    def cached_inputs(self, content, frames):
        from transformers.feature_extraction_utils import BatchFeature
        messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.past_ids is not None:
            text = '<|im_end|>\n' + text[text.index('<|im_start|>user'):]
        # Same placeholder expansion as Qwen2VLProcessor, using the saved native grids.
        parts = text.split(self.processor.video_token)
        expanded = [parts[0]]
        for frame, part in zip(frames, parts[1:]):
            expanded.extend([self.processor.video_token * len(frame['features']), part])
        expanded = ''.join(expanded)
        self.remember_input(expanded, content, frames)
        tokens = self.tokenizer([expanded], return_tensors='pt', return_attention_mask=False)
        inputs = BatchFeature({**tokens,
            'video_grid_thw': self.torch.stack([frame['grid'] for frame in frames]),
            # get_video_features below receives already projected features.
            'pixel_values_videos': self.torch.cat([frame['features'] for frame in frames]),
        }).to(self.device)
        if self.past_ids is not None:
            inputs['input_ids'] = self.torch.cat([self.past_ids, inputs.input_ids], dim=1)
        return inputs

    def window_inputs(self, current):
        from transformers.feature_extraction_utils import BatchFeature
        ids, frames = [self.header_ids], []
        for block in self.blocks:
            ids.extend([block['prompt_ids'], block['output_ids'], self.boundary_ids])
            frames.extend(block['frames'])
        ids.append(current['prompt_ids'])
        frames.extend(current['frames'])
        return BatchFeature({'input_ids': self.torch.cat(ids, dim=1),
            'video_grid_thw': self.torch.stack([frame['grid'] for frame in frames]),
            'pixel_values_videos': self.torch.cat([frame['features'] for frame in frames]),
        }).to(self.device)

    def generate(self, inputs, max_new_tokens, diagnosis=False):
        self.evicted_this_turn = 0
        current = self.current_block
        if inputs.input_ids.shape[1] + max_new_tokens > self.backend.context_limit:
            while True:
                if self.blocks and (current['frames'] or len(self.blocks) > 1):
                    removed = self.blocks.pop(0)
                    self.evicted_this_turn += len(removed['frames'])
                elif not self.blocks and len(current['frames']) > 1:
                    # Task 4 supplies all frames in one question; keep the most recent ones.
                    current['frames'].pop(0)
                    first_video = next(i for i, item in enumerate(current['content']) if item['type'] == 'video')
                    current['content'].pop(first_video)
                    self.cached_inputs(current['content'], current['frames'])
                    current = self.current_block
                    self.evicted_this_turn += 1
                else:
                    raise ValueError('System, user questions and newest observation exceed the context window.')
                token_count = self.header_ids.shape[1] + current['prompt_ids'].shape[1]
                token_count += sum(block['prompt_ids'].shape[1] + block['output_ids'].shape[1]
                                   + self.boundary_ids.shape[1] for block in self.blocks)
                if token_count + max_new_tokens <= self.backend.context_limit:
                    break
            self.evicted_frames += self.evicted_this_turn
            self.past = self.past_ids = None
            self.model.model.rope_deltas = None
            inputs = self.window_inputs(current)
        result = super().generate(inputs, max_new_tokens, diagnosis)
        # Retain exact generated pause/body tokens; the next ChatML prefix supplies EOS.
        current['output_ids'] = self.last_generated_ids[:, :-1].cpu()
        self.blocks.append(current)
        return {**result, 'visual_tokens': sum(len(frame['features']) for frame in current['frames']),
                **window_metadata(self)}

    def frame_inputs(self, frame, index, fps):
        content = [{'type': 'text', 'text': f'Time={index / fps:.1f}-{(index + 1) / fps:.1f}s'},
                   {'type': 'video'}]
        if index == 0:
            content.append({'type': 'text', 'text': self.backend.instruction})
        return self.cached_inputs(content, [frame])

    def observe(self, frame, index, fps, budget):
        return self.generate(self.frame_inputs(frame, index, fps), budget)

    def diagnose(self, question, budget):
        return self.generate(self.inputs([{"type": "text", "text": question}]), budget, diagnosis=True)


class Backend(LiveCC):
    def cached_video_features(self, pixel_values_videos, video_grid_thw):
        counts = (video_grid_thw.prod(-1) // self.processor.video_processor.merge_size**2).tolist()
        return self.torch.split(pixel_values_videos, counts)

    def __init__(self, config, device):
        super().__init__(config, device)
        self.model.model.get_video_features = self.cached_video_features

    def session(self, system="", observations=True):
        return Session(self, self.system + "\n" + system, observations)

    def answer_frames(self, frames, prompt, budget):
        session = self.session(observations=False)
        content = [{"type": "video"} for _ in frames] + [{"type": "text", "text": prompt}]
        return session.generate(session.cached_inputs(content, frames), budget, diagnosis=True)
