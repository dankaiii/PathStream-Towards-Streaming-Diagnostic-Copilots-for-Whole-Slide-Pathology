"""VideoLLM-online-8B-v1plus. Each frame is ten native projected visual tokens."""
import sys
from ...native import NativeAdapter, Runtime


NAME = "videollm_online"


SYSTEM = (
    "A multimodal AI assistant is helping users examine pathology views. "
    "Below is their conversation, interleaved with the list of video frames received by the assistant."
)


PROMPTS = {
    1: (
        'Observe the pathology views in their presented order. Briefly describe the morphology of '
        'each newly presented view in English. Describe the first frame. Remain silent while a '
        'similar view continues after your description. A finding may be described again in '
        'another view; do not restrict descriptions to facts absent from the entire history. Do '
        'not give a diagnosis.'
    ),
    2: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Remain '
        'silent while a similar view continues after your description. A finding may be described '
        'again in another view; do not restrict descriptions to facts absent from the entire '
        'history. Do not give a diagnosis.'
    ),
    3: (
        'Observe the pathology views in their presented order. Briefly describe the complete '
        'morphology of each newly presented view in English. Describe the first frame. Remain '
        'silent while a similar view continues after your description. A finding may be described '
        'again in another view; do not restrict descriptions to facts absent from the entire '
        'history. Give a diagnosis only when explicitly asked at the end.'
    ),
}


class VideoLLMOnline(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None):
        import torch

        sys.path.insert(0, str(config[NAME]['source_dir']))
        from models.live_llama import build_live_llama
        from models.modeling_live import fast_greedy_generate

        self.torch, self.config = torch, config
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.settings = config[NAME]
        self.system, self.instruction = system, instruction
        self.fast_generate = fast_greedy_generate
        self.model, self.tokenizer = build_live_llama(
            is_training=False, llm_pretrained=self.settings["base_model_path"],
            resume_from_checkpoint=self.settings["model_path"],
            vision_pretrained=self.settings["vision_model_path"],
            set_vision_inside=not config.get("preencoded_visual_features", False), frame_resolution=384, frame_token_cls=True,
            frame_token_pooled=[3, 3], frame_num_tokens=10, frame_token_interval=",",
            vision_hidden_size=1024, attn_implementation=self.settings["attention"],
            torch_dtype=torch.bfloat16,
        )
        self.model.to(self.device).eval().requires_grad_(False)
        self.model.config.use_cache = True
        self.context_limit = self.model.config.max_position_embeddings


class NativeSession:
    def __init__(self, backend, system):
        self.backend, self.model, self.tokenizer = backend, backend.model, backend.tokenizer
        self.torch, self.device = backend.torch, backend.device
        self.past = None
        self.token_count = 0
        self.pending = self.chat_ids([{"role": "system", "content": system}], add_stream_prompt=True)
        self.stream_prompt = self.chat_ids([{}], add_stream_prompt=True)
        self.generation_prompt = self.chat_ids([{}], add_stream_generation_prompt=True)
        self.interval_id = self.model.config.frame_token_interval_id
        self.eos_id = self.model.config.eos_token_id
        self.stream_open = True
        self.last_logits = None

    def chat_ids(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, return_tensors="pt", **kwargs).to(self.device)

    def ids(self, tokens):
        return self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)

    def reserve(self, additional):
        if self.token_count + additional > self.backend.context_limit:
            raise ValueError(f"VideoLLM-online context exceeds {self.backend.context_limit} tokens; "
                             "the native session is not truncated or reset.")

    def prefill(self, ids=None, embeds=None):
        if embeds is None:
            embeds = self.model.get_input_embeddings()(ids)
        self.reserve(embeds.shape[1])
        output = self.model(inputs_embeds=embeds, past_key_values=self.past,
                            use_cache=True, return_dict=True)
        self.past = output.past_key_values
        self.token_count += embeds.shape[1]
        self.last_logits = output.logits[:, -1]
        return output.logits

    def append_visual(self, visual):
        with self.torch.inference_mode():
            visual = visual.to(device=self.device, dtype=self.model.dtype).view(1, 10, -1)
            prefix = self.pending if self.stream_open else self.torch.cat(
                [self.pending, self.stream_prompt], dim=1)
            embeddings = self.torch.cat([self.model.get_input_embeddings()(prefix), visual], dim=1)
            self.prefill(embeds=embeddings)
        self.stream_open = True
        self.pending = self.ids([self.interval_id])

    def answer_prefix(self, query=None):
        if query is None:
            return self.generation_prompt
        prompt = self.chat_ids([{"role": "user", "content": query}],
                               add_stream_query_prompt=self.stream_open, add_generation_prompt=True)
        return prompt if self.stream_open else self.torch.cat([self.pending, prompt], dim=1)

    def answer(self, query, max_new_tokens, diagnosis=False):
        prefix = self.answer_prefix(query)
        input_tokens = self.token_count + prefix.shape[1]
        self.reserve(prefix.shape[1] + max_new_tokens)
        with self.torch.inference_mode():
            generated, self.past = self.backend.fast_generate(
                model=self.model, inputs_embeds=self.model.get_input_embeddings()(prefix),
                past_key_values=self.past, eos_token_id=self.eos_id,
                inplace_output_ids=self.torch.empty((1, max_new_tokens), device=self.device,
                                                   dtype=self.torch.long),
            )
        self.last_generated_ids = generated.clone()
        tokens = generated[0].tolist()
        self.token_count = input_tokens + len(tokens) - 1
        self.pending = generated[:, -1:].clone()
        self.stream_open = False
        stopped = tokens[-1] == self.eos_id
        text = self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()
        bad = not stopped or (diagnosis and not text)
        return {
            "response": text, "raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "decision": "protocol_error" if bad else "answer" if diagnosis else "update" if text else "silent",
            "protocol_error": bad, "input_tokens": input_tokens, "generated_tokens": len(tokens),
            "context_tokens_after_turn": self.token_count + 1, "stopped_on_eos": stopped,
            "hit_generation_limit": not stopped, "stop_reason": "eos" if stopped else "generation_limit",
            "visual_tokens": 10,
        }

    def observe(self, max_new_tokens):
        scores = self.last_logits.float().softmax(dim=-1)
        probability = scores[0, self.interval_id].item()
        threshold = self.backend.settings["frame_token_interval_threshold"]
        if probability < threshold:
            scores[0, self.interval_id] = 0
        decision = int(scores.argmax(dim=-1).item())
        if decision == self.interval_id:
            result = {
                "response": "", "raw_response": self.tokenizer.decode([decision]), "decision": "silent",
                "protocol_error": False, "input_tokens": self.token_count, "generated_tokens": 1,
                "context_tokens_after_turn": self.token_count + 1, "stopped_on_eos": False,
                "hit_generation_limit": False, "stop_reason": "native_frame_interval", "visual_tokens": 10,
            }
        elif decision == int(self.generation_prompt[0, 0]):
            result = self.answer(None, max_new_tokens)
        else:
            result = {
                "response": "", "raw_response": self.tokenizer.decode([decision]),
                "decision": "protocol_error", "protocol_error": True,
                "input_tokens": self.token_count, "generated_tokens": 1,
                "context_tokens_after_turn": self.token_count + 1, "stopped_on_eos": False,
                "hit_generation_limit": False, "stop_reason": "unexpected_native_stream_token", "visual_tokens": 10,
            }
        return {**result, "gate_token_id": decision, "gate_interval_probability": probability}


def window_metadata(session):
    return {'window_frames': sum(len(block['frames']) for block in session.blocks),
            'evicted_frames_this_turn': session.evicted_this_turn,
            'evicted_frames_total': session.evicted_frames,
            'max_context_tokens': session.backend.context_limit,
            'kv_cache_reset_on_window_eviction': bool(session.evicted_this_turn)}


class WindowedVideoSession(NativeSession):
    def __init__(self, backend, system, observations=True):
        super().__init__(backend, system)
        self.system = system
        self.observations = observations
        self.blocks = []
        self.evicted_frames = self.evicted_this_turn = 0
        messages = [{'role': 'system', 'content': system}]
        if observations:
            messages.append({'role': 'user', 'content': backend.instruction})
        self.window_prefix = self.chat_ids(messages, add_stream_prompt=True)

    def state(self):
        return self.token_count, self.pending.shape[1], self.stream_open

    def retained_state(self):
        count, pending, opened = 0, self.window_prefix.shape[1], True
        for block in self.blocks:
            count += pending + (0 if opened else self.stream_prompt.shape[1]) + 10
            tokens = block['output_ids']
            if tokens is not None:
                count += self.generation_prompt.shape[1] + tokens.shape[1] - 1
            pending, opened = 1, tokens is None
        return count, pending, opened

    def make_room(self, required, keep_latest):
        if required(*self.state()) <= self.backend.context_limit:
            return
        removed = 0
        while True:
            if len(self.blocks) <= int(keep_latest):
                raise ValueError('System, user questions and newest observation exceed the context window.')
            self.blocks.pop(0)
            removed += 1
            if required(*self.retained_state()) <= self.backend.context_limit:
                break
        self.evicted_frames += removed
        self.evicted_this_turn += removed
        self.rebuild()

    def rebuild(self):
        # Replay retained observations and exact previous outputs; never generate them again.
        NativeSession.__init__(self, self.backend, self.system)
        self.pending = self.window_prefix
        with self.torch.inference_mode():
            for block in self.blocks:
                NativeSession.append_visual(self, block['frames'][0])
                tokens = block['output_ids']
                if tokens is not None:
                    tokens = tokens.to(self.device)
                    self.prefill(ids=self.generation_prompt)
                    if tokens.shape[1] > 1:
                        self.prefill(ids=tokens[:, :-1])
                    self.pending = tokens[:, -1:].clone()
                    self.stream_open = False

    def append_visual(self, visual):
        self.evicted_this_turn = 0
        first = not self.blocks and self.evicted_frames == 0
        query_tokens = (self.chat_ids([{'role': 'user', 'content': self.backend.instruction}],
                         add_stream_query_prompt=True, add_generation_prompt=True).shape[1]
                        if first and self.observations else self.generation_prompt.shape[1])
        budget = self.backend.config['description_max_new_tokens' if self.observations else 'answer_max_new_tokens']
        def required(count, pending, opened):
            return count + pending + (0 if opened else self.stream_prompt.shape[1]) + 10 + query_tokens + budget
        self.make_room(required, keep_latest=False)
        super().append_visual(visual)
        self.blocks.append({'frames': [visual], 'output_ids': None})

    def answer(self, query, max_new_tokens, diagnosis=False):
        if diagnosis:
            self.evicted_this_turn = 0
        prefixes = ({opened: self.chat_ids([{'role': 'user', 'content': query}],
                     add_stream_query_prompt=opened, add_generation_prompt=True).shape[1]
                     for opened in (False, True)} if query is not None else None)
        def required(count, pending, opened):
            prefix = (prefixes[opened] + (0 if opened else pending)
                      if prefixes is not None else self.generation_prompt.shape[1])
            return count + prefix + max_new_tokens
        self.make_room(required, keep_latest=True)
        result = super().answer(query, max_new_tokens, diagnosis)
        if not diagnosis:
            self.blocks[-1]['output_ids'] = self.last_generated_ids.cpu()
        return {**result, **window_metadata(self)}

    def observe(self, max_new_tokens):
        result = super().observe(max_new_tokens)
        return {**result, **window_metadata(self)}


class Session(WindowedVideoSession):
    def observe(self, frame, index, fps, budget):
        self.append_visual(frame)
        if index == 0:
            return self.answer(self.backend.instruction, budget)
        return super().observe(budget)

    def diagnose(self, question, budget):
        return self.answer(question, budget, diagnosis=True)


class Backend(VideoLLMOnline):
    def session(self, system="", observations=True):
        return Session(self, self.system + "\n" + system, observations)

    def answer_frames(self, frames, prompt, budget):
        session = self.session(observations=False)
        for frame in frames:
            session.append_visual(frame)
        return session.answer(prompt, budget, diagnosis=True)


DEFAULTS = {
    "frame_token_interval_threshold": 0.0,
    "attention": 'flash_attention_2',
}


def load(model_path, base_model_path, vision_model_path, source_dir, *, device="cuda:0", **settings):
    options = {**DEFAULTS, **settings, "model_path": model_path}
    options['source_dir'] = source_dir
    options.update(base_model_path=base_model_path, vision_model_path=vision_model_path)
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": 256, "diagnosis_max_new_tokens": 256, "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    adapter = NativeAdapter(backend, PROMPTS)
    return adapter
