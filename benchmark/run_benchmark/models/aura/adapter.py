"""AURA. Use its native vLLM gate, two-tier context and cross-turn penalty."""
import asyncio
import os
from .history import CrossTurnPenalty, SessionHistory, SILENT_TEXT
from ...native import NativeAdapter, Runtime


NAME = "aura"


SYSTEM = (
    "You are receiving a live pathology video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output '<|silent|>' to signify silence. Respond in English."
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


SILENT_TOKEN_ID = 151669


IM_END_TOKEN_ID = 151645


VIDEO_PAD_TOKEN_ID = 151656


HISTORY_SETTINGS = {
    "max_rounds": 45, "num_rounds_keep": 30, "pruning_enabled": True,
    "max_context_qas": 10, "max_1qna_rounds": 4,
}


ENGINE_SETTINGS = {
    "trust_remote_code": False, "enable_prefix_caching": True,
    "kv_offloading_size": 10, "disable_hybrid_kv_cache_manager": True,
    "block_size": 16, "prefix_caching_hash_algo": "xxhash",
    "mm_encoder_attn_backend": "FLASH_ATTN", "mm_encoder_tp_mode": "data",
    "max_num_batched_tokens": 15360, "cudagraph_capture_sizes": [1, 2, 4],
    # Frames are already selected by the benchmark. Do not resample or inspect future frames.
    "mm_processor_kwargs": {"do_sample_frames": False},
}


class AURA(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None):
        self.config, self.settings = config, config[NAME]
        self.system, self.instruction = system, instruction
        first = int(device.split(":")[1])
        size = self.settings["tensor_parallel_size"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = (visible.split(",")[first:first + size] if visible is not None
                   else [str(i) for i in range(first, first + size)])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.loop = asyncio.new_event_loop()
        self.engine = self.loop.run_until_complete(self._create_engine())
        self.tokenizer = self.engine.get_tokenizer()
        self.context_limit = self.settings["max_model_len"]
        self.request_count = 0

    async def _create_engine(self):
        from vllm.config import AttentionConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        settings = self.settings
        return AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=settings["model_path"],
            tensor_parallel_size=settings["tensor_parallel_size"],
            max_model_len=settings["max_model_len"],
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            attention_config=AttentionConfig(backend="FLASH_ATTN"),
            enable_mm_embeds=self.config.get("preencoded_visual_features", False),
            **ENGINE_SETTINGS,
        ))

    async def _generate(self, prompt, params, native_gate):
        self.request_count += 1
        request_id = f"aura-{self.request_count}"
        stream = self.engine.generate(prompt, params, request_id)
        async for result in stream:
            output = result.outputs[0]
            if native_gate and output.token_ids and output.token_ids[0] in (
                    SILENT_TOKEN_ID, IM_END_TOKEN_ID):
                await self.engine.abort(request_id)
                await stream.aclose()
                return result, True
        return result, False

    def generate(self, prompt, params, native_gate=False):
        return self.loop.run_until_complete(self._generate(prompt, params, native_gate))

    def close(self):
        self.engine.shutdown()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()


class NativeSession:
    def answer(self, max_new_tokens, diagnosis=False):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.backend.settings["temperature"], max_tokens=max_new_tokens,
            **self.penalty.build_sampling_kwargs(),
        )
        result, silent = self.backend.generate(
            self.history.get_vllm_inputs(), params, native_gate=True)
        output = result.outputs[0]
        tokens = list(output.token_ids)
        text = "" if silent else output.text
        history_text = SILENT_TEXT if silent else text
        self.history.add_assistant_message(history_text)
        self.penalty.record(None if silent else text)
        input_tokens = len(result.prompt_token_ids)
        # Every saved assistant message has an IM_END delimiter in the native grammar.
        self.token_count = input_tokens + len(self.tokenizer.encode(
            history_text, add_special_tokens=False)) + 1
        bad = not silent and not text.strip()
        if diagnosis and not text.strip():
            bad = True
        return {
            "response": text.strip(),
            "raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "decision": "protocol_error" if bad else "answer" if diagnosis else "silent" if silent else "update",
            "protocol_error": bad, "input_tokens": input_tokens,
            "generated_tokens": len(tokens), "context_tokens_after_turn": self.token_count,
            "stopped_on_eos": bool(tokens and tokens[-1] in (IM_END_TOKEN_ID, 151643)),
            "hit_generation_limit": not silent and output.finish_reason == "length",
            "stop_reason": "native_silent_token" if silent else output.finish_reason,
            "native_silence": silent, "first_token_id": tokens[0],
            "visual_tokens": result.prompt_token_ids.count(VIDEO_PAD_TOKEN_ID),
            "num_cached_tokens": result.num_cached_tokens,
            "sliding_window_rounds": self.history._sw_round_count(),
            "context_history_qas": len(self.history._context_history),
        }


def window_result(session, frames):
    return {'window_frames': frames, 'evicted_frames_this_turn': session.evicted_this_turn,
            'evicted_frames_total': session.evicted_frames,
            'max_context_tokens': session.backend.context_limit}


class CachedHistory(SessionHistory):
    def __init__(self):
        self.pinned_questions = []
        super().__init__(**HISTORY_SETTINGS)

    def _rebuild_history(self):
        super()._rebuild_history()
        self.history[1:1] = self.pinned_questions

    def videos(self):
        return [item['video'] for msg in self.history if isinstance(msg['content'], list)
                for item in msg['content'] if item['type'] == 'video']

    def get_vllm_inputs(self):
        import torch
        prompt = super().get_vllm_inputs()
        videos = self.videos()
        if videos:
            prompt['multi_modal_data'] = {'video': {
                'video_embeds': torch.cat([frame['features'] for frame, _ in videos]),
                'video_grid_thw': torch.stack([frame['grid'] for frame, _ in videos]),
                'timestamps': [[metadata['timestamp']] for _, metadata in videos],
            }}
        return prompt

    def input_length(self, tokenizer):
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
        prompt = super().get_vllm_inputs()['prompt']
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        for frame, metadata in self.videos():
            replacement = Qwen3VLMultiModalProcessor.get_video_repl(
                tokens_per_frame=[len(frame['features'])], timestamps=[metadata['timestamp']],
                tokenizer=tokenizer, vision_start_token_id=151652,
                vision_end_token_id=151653, video_token_id=151656, select_token_id=True)
            length += len(replacement.full) - 3
        return length

    def evict_oldest(self):
        if self._context_history:
            oldest = self._context_history.pop(0)
            self.pinned_questions.extend(msg for msg in oldest
                if msg['role'] == 'user' and self._has_user_text(msg))
            self._rebuild_history()
            return 0
        if len(self.videos()) <= 1:
            raise ValueError('System, user questions and newest observation exceed the context window.')
        index = next(i for i, msg in enumerate(self._sliding_window)
                     if isinstance(msg['content'], list)
                     and any(item['type'] == 'video' for item in msg['content']))
        user = self._sliding_window[index]
        content = user['content']
        content.pop(next(i for i, item in enumerate(content) if item['type'] == 'video'))
        if not any(item['type'] == 'video' for item in content):
            if self._has_user_text(user):
                self.pinned_questions.append({'role': 'user', 'content': self._extract_user_text(content)})
            self._sliding_window.pop(index)
            if index < len(self._sliding_window) and self._sliding_window[index]['role'] == 'assistant':
                self._sliding_window.pop(index)
        self.current_rounds = self._sw_round_count()
        self._rebuild_history()
        return 1


class Session(NativeSession):
    def __init__(self, backend, system):
        self.backend, self.tokenizer = backend, backend.tokenizer
        self.history = CachedHistory()
        self.history.system_prompt = system
        self.history._system_msg = {'role': 'system', 'content': system}
        self.history._rebuild_history()
        self.penalty = CrossTurnPenalty(self.tokenizer)
        self.token_count = 0
        self.evicted_frames = self.evicted_this_turn = 0

    def answer(self, max_new_tokens, diagnosis=False):
        self.evicted_this_turn = 0
        while self.history.input_length(self.tokenizer) + max_new_tokens > self.backend.context_limit:
            self.evicted_this_turn += self.history.evict_oldest()
        self.evicted_frames += self.evicted_this_turn
        result = super().answer(max_new_tokens, diagnosis)
        if result['hit_generation_limit']:
            result.update(protocol_error=True, decision='protocol_error', stop_reason='generation_limit')
        return {**result, **window_result(self, len(self.history.videos()))}

    def observe(self, frame, index, fps, budget):
        self.history.add_user_message(self.backend.instruction if index == 0 else '',
                                     video_tuple=(frame, {'timestamp': index / fps}))
        return self.answer(budget)

    def diagnose(self, question, budget):
        self.history.add_user_message(question)
        return self.answer(budget, diagnosis=True)


class Backend(AURA):
    def session(self, tissue_context=''):
        return Session(self, self.system + '\n' + tissue_context)

    def answer_frames(self, frames, prompt, budget):
        session = self.session()
        session.history.add_user_message(prompt)
        # Task 4 is a single public question with all selected visual observations.
        content = session.history._sliding_window[-1]['content']
        content[:0] = [{'type': 'video', 'video': (frame, {'timestamp': 0.0})} for frame in frames]
        return session.answer(budget, diagnosis=True)


DEFAULTS = {
    "temperature": 0.5,
    "tensor_parallel_size": 1,
    "max_model_len": 262144,
    "gpu_memory_utilization": 0.9,
    "max_tokens": 128,
}


def load(model_path, *, device="cuda:0", **settings):
    options = {**DEFAULTS, **settings, "model_path": model_path}
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": options['max_tokens'], "diagnosis_max_new_tokens": options['max_tokens'], "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    adapter = NativeAdapter(backend, PROMPTS)
    return adapter
