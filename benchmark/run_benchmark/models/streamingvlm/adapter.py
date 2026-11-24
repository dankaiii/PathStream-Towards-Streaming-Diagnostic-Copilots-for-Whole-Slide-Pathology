"""StreamingVLM. Use the native 16-round shrink cache and streaming chunk decoder."""
import sys
from ...native import NativeAdapter, Runtime


NAME = "streamingvlm"


SYSTEM = "You are an expert pathologist observing a stream of pathology views."


PROMPTS = {
    1: (
        'Observe the pathology views in their presented order. Briefly describe the morphology of '
        'each newly presented view in English. Describe the first frame. Remain silent while a '
        'similar view continues after your description. A finding may be described again in '
        'another view; do not restrict descriptions to facts absent from the entire history. Do '
        'not give a diagnosis. Keep the native streaming format with the trailing ... waiting '
        'marker; the waiting marker alone is sufficient when there is nothing to describe.'
    ),
    2: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Wait while '
        'a similar view continues after your description. A finding may be described again in '
        'another view. Keep the native streaming format with the trailing ... waiting marker; the '
        'waiting marker alone is sufficient when there is nothing to describe. Do not give a '
        'diagnosis.'
    ),
    3: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Wait while '
        'a similar view continues after your description. A finding may be described again in '
        'another view. Keep the native streaming format with the trailing ... waiting marker; the '
        'waiting marker alone is sufficient when there is nothing to describe. Give a diagnosis '
        'only when explicitly asked at the end.'
    ),
}


WINDOW_ROUNDS = 16


TEXT_SINK = 512


TEXT_WINDOW = 512


MIN_PIXELS = 128 * 28 * 28


MAX_PIXELS = 768 * 28 * 28


def patch_streaming_model(model):
    """Bind the official Qwen2.5 runtime without importing demo/training side effects."""
    from types import MethodType
    from streaming_vlm.inference.qwen2_5.vision_forward import (
        streaming_visual_attention_forward, streaming_visual_block_forward,
        streaming_visual_encoder_forward,
    )
    from streaming_vlm.inference.qwen2_5.language_forward import (
        streaming_language_model_forward, streaming_text_flash_attn_forward,
        streaming_text_decoder_layer_forward, _update_causal_mask,
    )
    from streaming_vlm.inference.qwen2_5.model_forward import model_forward, qwen2_5_vl_forward
    from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index
    from streaming_vlm.inference.generate.streaming_generate_qwen import streaming_generate, _sample
    from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation

    model.generate = MethodType(streaming_generate, model)
    model._sample = MethodType(_sample, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model.forward = MethodType(qwen2_5_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    language = model.model.language_model
    language.forward = MethodType(streaming_language_model_forward, language)
    language._update_causal_mask = MethodType(_update_causal_mask, language)
    for layer in language.layers:
        layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
        layer.self_attn.forward = MethodType(streaming_text_flash_attn_forward, layer.self_attn)
    visual = model.model.visual
    visual.forward = MethodType(streaming_visual_encoder_forward, visual)
    for block in visual.blocks:
        block.forward = MethodType(streaming_visual_block_forward, block)
        block.attn.forward = MethodType(streaming_visual_attention_forward, block.attn)
    model.model.get_rope_index = MethodType(get_rope_index, model.model)


class StreamingVLM(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None, streaming=True):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

        sys.path.insert(0, str(config[NAME]['source_dir']))
        self.torch, self.config = torch, config
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.settings = config[NAME]
        self.system, self.instruction = system, instruction
        set_seed(self.settings["seed"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.settings["model_path"], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": device}, attn_implementation="flash_attention_2",
        ).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            self.settings["model_path"], local_files_only=True, use_fast=False,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        self.tokenizer = self.processor.tokenizer
        if streaming:
            patch_streaming_model(self.model)
        self.context_limit = self.model.config.max_position_embeddings


class NativeSession:
    def __init__(self, backend, system):
        from streaming_vlm.inference.streaming_args import StreamingArgs
        from streaming_vlm.inference.generate.streaming_cache import StreamingCache
        from streaming_vlm.utils.get_qwen_range import get_qwen_range

        self.backend, self.model, self.processor = backend, backend.model, backend.processor
        self.torch, self.device = backend.torch, backend.device
        self.tokenizer, self.system = backend.tokenizer, system
        self.streaming = StreamingArgs(pos_mode="shrink", all_text=False)
        self.past = StreamingCache()
        self.tokens = None
        self.round = 0
        self.get_range = get_qwen_range
        self.wait_ids = self.encode(" ...")
        self.assistant_header_length = len(self.encode("<|im_start|>assistant\n"))
        self.grids = []
        self.seconds = []

    @property
    def token_count(self):
        return 0 if self.tokens is None else self.tokens.shape[1]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tensor(self, ids):
        return self.torch.tensor([ids], dtype=self.torch.long, device=self.device)

    def reindex(self, order):
        """Move/delete exactly the same positions in token IDs and unrotated KV."""
        cached = self.past.get_seq_length()
        self.tokens = self.tokens[:, order].contiguous()
        kv_order = self.torch.tensor([i for i in order if i < cached], device=self.device)
        for layer in range(len(self.past.key_cache)):
            self.past.key_cache[layer] = self.past.key_cache[layer].index_select(2, kv_order).contiguous()
            self.past.value_cache[layer] = self.past.value_cache[layer].index_select(2, kv_order).contiguous()
        self.past._seen_tokens = self.past.get_seq_length()

    def remove(self, start, end):
        self.reindex(list(range(start)) + list(range(end + 1, self.token_count)))

    def prune_native_history(self):
        # Official process_past_kv with visual_round == text_round == 16.
        # Move the oldest assistant body into "previous text", then remove that
        # user/assistant round. No summaries, re-encoding, or additional eviction.
        if self.round >= WINDOW_ROUNDS:
            start, end = self.get_range(self.tokens, "assistant", 0, contain_lf=False)
            body_start, body_end = start + self.assistant_header_length, end
            # Upstream removes the " ..." trailer. Preserve real body tokens when
            # the native 20-token budget ended before the trailer was generated.
            if self.tokens[0, body_end - len(self.wait_ids):body_end].tolist() == self.wait_ids:
                body_end -= len(self.wait_ids)
            _, previous_end = self.get_range(self.tokens, "previous text", 0, contain_lf=False)
            if body_start < body_end:
                self.reindex(list(range(previous_end)) + list(range(body_start, body_end)) +
                             list(range(previous_end, body_start)) +
                             list(range(body_end, self.token_count)))
            self.remove(*self.get_range(self.tokens, "assistant", 0))
            self.remove(*self.get_range(self.tokens, "user", 0))
            self.grids.pop(0)
            self.seconds.pop(0)
        if self.round:
            start, end = self.get_range(self.tokens, "previous text", 0)
            cut_start, cut_end = start + TEXT_SINK + 4, end - TEXT_WINDOW - 1
            if cut_start <= cut_end:
                self.remove(cut_start, cut_end)

    def render(self, message, initial=False):
        prefix = [{"role": "system", "content": self.system}]
        if initial:
            return self.processor.apply_chat_template(
                prefix + [{"role": "previous text", "content": ""}, message],
                tokenize=False, add_generation_prompt=True,
            )
        # Derive the system boundary from this template, not upstream's hardcoded
        # 58-character offset, because the pathology system prompt is different.
        system_text = self.processor.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False)
        text = self.processor.apply_chat_template(prefix + [message], tokenize=False, add_generation_prompt=True)
        return "\n" + text[len(system_text):]

    def append_inputs(self, inputs):
        if self.tokens is not None:
            new_ids = inputs["input_ids"]
            if self.tokens[0, -1].item() == 198:
                new_ids = new_ids[:, 1:]
            inputs["input_ids"] = self.torch.cat([self.tokens, new_ids], dim=1)
        inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])
        self.streaming.input_ids = inputs["input_ids"]
        self.streaming.video_grid_thw = self.torch.cat(self.grids, dim=0)
        self.streaming.second_per_grid_ts = self.torch.cat(self.seconds, dim=0)
        return inputs

    def generate(self, inputs, max_new_tokens, diagnosis=False):
        input_length = inputs["input_ids"].shape[1]
        visual_tokens = int((inputs["input_ids"] == self.model.config.video_token_id).sum().item())
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, past_key_values=self.past, streaming_args=self.streaming,
                max_new_tokens=max_new_tokens, use_cache=True, return_dict_in_generate=True,
                do_sample=True, temperature=0.9, repetition_penalty=1.05, pad_token_id=151645,
            )
        generated = output.sequences[:, input_length:]
        stopped = generated[0, -1].item() in self.model.generation_config.eos_token_id
        raw = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        self.tokens, self.past = output.sequences, output.past_key_values
        # Same turn closure as upstream; any appended EOS is still uncached.
        if self.tokens[0, -1].item() != 151645:
            self.tokens = self.torch.cat([self.tokens, self.tensor([151645])], dim=1)
        body = raw.strip()
        if body.endswith("..."):
            body = body[:-3].rstrip()
        bad = diagnosis and (not body or not stopped)
        result = {
            "response": body, "raw_response": raw,
            "decision": "protocol_error" if bad else "answer" if diagnosis else "update" if body else "silent",
            "protocol_error": bad, "hit_generation_limit": not stopped,
            "input_tokens": input_length, "generated_tokens": generated.shape[1],
            "context_tokens_after_turn": self.token_count, "visual_tokens": visual_tokens,
            "native_wait_marker": raw.rstrip().endswith("..."),
            "native_chunk_budget": not diagnosis,
        }
        self.torch.cuda.synchronize(self.device)
        return result

    def frame(self, image, index, fps):
        inputs = self.frame_inputs(image, index, fps)
        result = self.generate(inputs, self.backend.settings["stream_max_new_tokens"])
        self.round += 1
        return result

    def diagnose(self, question, max_new_tokens):
        return self.generate(self.question_inputs(question), max_new_tokens, diagnosis=True)


def window_result(session, frames):
    return {'window_frames': frames, 'evicted_frames_this_turn': session.evicted_this_turn,
            'evicted_frames_total': session.evicted_frames,
            'max_context_tokens': session.backend.context_limit}


class Session(NativeSession):
    def __init__(self, backend, system):
        super().__init__(backend, system)
        self.evicted_frames = self.evicted_this_turn = 0

    def visual_segment(self, content, frames, fps, initial):
        from transformers.feature_extraction_utils import BatchFeature
        text = self.render({'role': 'user', 'content': content}, initial=initial)
        parts = text.split(self.processor.video_token)
        text = parts[0]
        for frame, tail in zip(frames, parts[1:]):
            text += self.processor.video_token * len(frame['features']) + tail
        tokens = self.tokenizer([text], add_special_tokens=False, return_tensors='pt')
        return BatchFeature({**tokens,
            'pixel_values_videos': self.torch.cat([frame['features'] for frame in frames]),
            'video_grid_thw': self.torch.stack([frame['grid'] for frame in frames]),
            'second_per_grid_ts': self.torch.tensor([2.0 / fps] * len(frames)),
        }).to(self.device)

    def frame_inputs(self, frame, index, fps):
        # Keep upstream's visual eviction, text sink and recent-text compression.
        self.prune_native_history()
        content = [{'type': 'text', 'text': f'Time={index / fps:.1f}-{(index + 1) / fps:.1f}s'},
                   {'type': 'video', 'video': 'cached_benchmark_frame'}]
        if index == 0:
            content.append({'type': 'text', 'text': self.backend.instruction})
        self.current_segment = self.visual_segment(content, [frame], fps, initial=index == 0)
        self.grids.append(self.current_segment['video_grid_thw'])
        self.seconds.append(self.current_segment['second_per_grid_ts'])
        return self.append_inputs(dict(self.current_segment))

    def question_inputs(self, question):
        text = self.render({'role': 'user', 'content': [{'type': 'text', 'text': question}]})
        self.current_segment = self.tokenizer(text, add_special_tokens=False,
                                              return_tensors='pt').to(self.device)
        return self.append_inputs(dict(self.current_segment))

    def generate(self, inputs, max_new_tokens, diagnosis=False):
        self.evicted_this_turn = 0
        while inputs['input_ids'].shape[1] + max_new_tokens > self.backend.context_limit:
            if self.tokens is None or len(self.grids) <= 1:
                raise ValueError('System, user questions and newest observation exceed the context window.')
            # Same unrotated-KV relocation as upstream, without adding this
            # budget-evicted answer to the compressed text history.
            self.remove(*self.get_range(self.tokens, 'assistant', 0))
            self.remove(*self.get_range(self.tokens, 'user', 0))
            self.grids.pop(0)
            self.seconds.pop(0)
            self.evicted_this_turn += 1
            inputs = self.append_inputs(dict(self.current_segment))
        self.evicted_frames += self.evicted_this_turn
        result = super().generate(inputs, max_new_tokens, diagnosis)
        return {**result, **window_result(self, len(self.grids)),
                'stop_reason': ('generation_limit' if diagnosis else 'native_chunk_budget')
                               if result['hit_generation_limit'] else 'eos',
                'kv_cache_policy': 'native_unrotated_KV_relocation_and_shrink_positions'}

    def observe(self, frame, index, fps, budget):
        return self.frame(frame, index, fps)

    def answer_frames(self, frames, prompt, budget):
        # One visual question, without producing intermediate descriptions.
        frames = list(frames)
        removed = 0
        while True:
            content = [{'type': 'video', 'video': 'cached_benchmark_frame'} for _ in frames]
            content.append({'type': 'text', 'text': prompt})
            self.current_segment = self.visual_segment(content, frames, 1.0, initial=True)
            if self.current_segment['input_ids'].shape[1] + budget <= self.backend.context_limit:
                break
            if len(frames) <= 1:
                raise ValueError('Question and newest observation exceed the context window.')
            frames.pop(0)
            removed += 1
        self.evicted_frames += removed
        self.grids = list(self.current_segment['video_grid_thw'].split(1))
        self.seconds = list(self.current_segment['second_per_grid_ts'].split(1))
        result = self.generate(self.append_inputs(dict(self.current_segment)), budget, diagnosis=True)
        return {**result, 'evicted_frames_this_turn': removed}


class Backend(StreamingVLM):
    def __init__(self, config, device, instruction=None):
        super().__init__(config, device, instruction=instruction)
        self.model.model.get_video_features = lambda features, grids: features

    def session(self, tissue_context=''):
        return Session(self, self.system + '\n' + tissue_context)

    def answer_frames(self, frames, prompt, budget):
        return self.session().answer_frames(frames, prompt, budget)


DEFAULTS = {
    "seed": 42,
    "stream_max_new_tokens": 20,
}


def load(model_path, source_dir, *, device="cuda:0", **settings):
    options = {**DEFAULTS, **settings, "model_path": model_path}
    options['source_dir'] = source_dir
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": options['stream_max_new_tokens'], "diagnosis_max_new_tokens": 256, "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    adapter = NativeAdapter(backend, PROMPTS)
    return adapter
