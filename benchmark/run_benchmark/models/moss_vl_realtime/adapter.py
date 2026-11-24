"""MOSS-VL-Realtime. Submit one observation per native idle boundary."""
from queue import Queue
from threading import Thread
from ...native import NativeAdapter, Runtime


NAME = "moss_vl_realtime"


SYSTEM = (
    "You are a helpful AI assistant specializing in real-time pathology observation. "
    "Frames arrive one at a time. Decide whether to describe the current view or wait. "
    "Finish each response with <|silence|>; output <|silence|> alone when waiting. "
    "Wait for visual input before describing anything."
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


class MossVLRealtime(Runtime):
    def __init__(self, config, device, system=SYSTEM, instruction=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch, self.config, self.settings = torch, config, config[NAME]
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.system, self.instruction = system, instruction
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings["model_path"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map=self.settings.get("device_map", {"": device}),
            attn_implementation="flash_attention_2",
        ).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            self.settings["model_path"], trust_remote_code=True, local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.context_limit = self.model.config.text_config.max_position_embeddings
        self.active_session = None

    def close_session(self):
        if self.active_session is not None:
            self.active_session.close()
            self.active_session = None


class NativeTokenStream:
    def __init__(self, session):
        self.session = session
        self.prepare(256)

    def prepare(self, limit):
        self.limit, self.count, self.input_length = limit, 0, 0

    def __call__(self, input_ids, scores):
        self.input_length = input_ids.shape[1]
        return scores

    def put(self, token):
        value = int(token.item())
        self.count += 1
        # Upstream ignores HF stopping criteria in realtime mode. A benchmark
        # generation limit terminates this sample; it never becomes a wait decision.
        if self.count >= self.limit and value != self.session.silence_id:
            self.session.model.stop_real_time_generate()
        self.session.generated.put(("token", value, self.input_length))

    def end(self):
        pass


class NativeSession:
    def __init__(self, backend, system):
        self.backend, self.model, self.processor = backend, backend.model, backend.processor
        self.tokenizer, self.torch, self.device = backend.tokenizer, backend.torch, backend.device
        self.system = system
        self.silence_id = self.tokenizer.convert_tokens_to_ids("<|silence|>")
        self.frames, self.prompts, self.generated = Queue(), Queue(), Queue()
        self.token_count = 0
        self.streamer = NativeTokenStream(self)
        self.thread = Thread(target=self.run, name="moss-benchmark-realtime", daemon=True)
        self.thread.start()
        # No visual frame has been sent here. Native initialization is excluded
        # from frame decisions and from reference body likelihood.
        bootstrap = self.read_turn()
        if bootstrap["protocol_error"]:
            self.close()
            raise RuntimeError("MOSS realtime initialization did not reach its native idle token")

    def read_turn(self, diagnosis=False):
        tokens = []
        first_input_length = None
        while True:
            kind, value, input_length = self.generated.get()
            if kind == "error":
                raise value
            if kind == "end":
                raise RuntimeError("MOSS realtime decoder stopped before completing the current turn")
            if first_input_length is None:
                first_input_length = input_length
            tokens.append(value)
            self.token_count = input_length + 1
            if value == self.silence_id or len(tokens) >= self.streamer.limit:
                break
        stopped = tokens[-1] == self.silence_id
        body = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        bad = not stopped
        self.torch.cuda.synchronize(self.device)
        return {
            "response": body, "raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False),
            "decision": "protocol_error" if bad else "answer" if diagnosis else "update" if body else "silent",
            "protocol_error": bad, "input_tokens": first_input_length, "generated_tokens": len(tokens),
            "context_tokens_after_turn": self.token_count, "visual_tokens": None,
            "native_wait_marker": stopped, "stopped_on_eos": False,
            "hit_generation_limit": not stopped,
            "stop_reason": "native_silence" if stopped else "generation_limit",
        }

    def frame(self, image, index, fps, max_new_tokens):
        self.streamer.prepare(max_new_tokens)
        self.frames.put((image, index / fps))
        return self.read_turn()

    def diagnose(self, question, max_new_tokens):
        self.streamer.prepare(max_new_tokens)
        self.prompts.put(question)
        return self.read_turn(diagnosis=True)

    def close(self):
        self.model.stop_real_time_generate()
        self.thread.join()


class CachedProcessor:
    def __init__(self, session):
        self.session = session
        self.tokenizer = session.tokenizer

    def __call__(self, text, images, **kwargs):
        from transformers.feature_extraction_utils import BatchFeature
        session = self.session
        session.visible_frames.extend(images)
        tokens = self.tokenizer(text.replace('<|image|>', '<|image_pad|>'), **kwargs)
        return BatchFeature({**tokens,
            'pixel_values': session.torch.cat([frame['features'] for frame in images]),
            'grid_thw': session.torch.stack([frame['grid'] for frame in images]),
            'media_nums_per_sample': [len(images)],
        })


class Session(NativeSession):
    def __init__(self, backend, system, instruction=None):
        self.visible_frames = []
        self.initial_instruction = backend.instruction if instruction is None else instruction
        backend.active_session = self
        super().__init__(backend, system)

    def run(self):
        from transformers import LogitsProcessorList
        try:
            self.torch.cuda.set_device(self.device)
            messages = [{'role': 'system', 'content': self.system},
                        {'role': 'user', 'content': self.initial_instruction}]
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_tensors='pt').to(self.device)
            length = ids.shape[1]
            positions = self.torch.arange(length, device=self.device).view(1, 1, length).expand(3, 1, length)
            self.model.start_real_time_generate()
            with self.torch.inference_mode():
                self.model._real_time_generate(
                    new_video_frames=self.frames, new_prompts=self.prompts, output_text_queue=None,
                    processor=CachedProcessor(self), inputs=ids, attention_mask=self.torch.ones_like(ids),
                    position_ids=positions.contiguous(), realtime_next_position=length,
                    full_vision_token_info=None, max_tokens_per_turn=86400,
                    do_sample=False, repetition_penalty=1.0, use_cache=True,
                    streamer=self.streamer, logits_processor=LogitsProcessorList([self.streamer]))
        except BaseException as error:
            self.generated.put(('error', error, None))
        finally:
            self.model.stop_real_time_generate()
            self.generated.put(('end', None, None))

    def observe(self, frame, index, fps, budget):
        result = self.frame(frame, index, fps, budget)
        return {**result, 'visual_tokens': len(frame['features']) + 1}

    def answer_frames(self, frames, budget):
        self.streamer.prepare(budget)
        # Submit a single atomic queue batch, so decoding starts after all Task 4 frames.
        with self.frames.mutex:
            self.frames.queue.extend((frame, float(index)) for index, frame in enumerate(frames))
            self.frames.not_empty.notify()
        return self.read_turn(diagnosis=True)


class Backend(MossVLRealtime):
    def __init__(self, config, device, instruction=None):
        super().__init__(config, device, instruction=instruction)
        self.model.model.visual.forward = lambda hidden_states, grid_thw, **kwargs: hidden_states

    def session(self, tissue_context='', instruction=None):
        self.close_session()
        return Session(self, self.system + '\n' + tissue_context, instruction=instruction)

    def answer_frames(self, frames, prompt, budget):
        return self.session(instruction=prompt).answer_frames(frames, budget)

    def end_case(self):
        self.close_session()


def load(model_path, *, device="cuda:0", **settings):
    options = {**settings, "model_path": model_path}
    config = {NAME: options, "preencoded_visual_features": True,
              "description_max_new_tokens": 256, "diagnosis_max_new_tokens": 256, "answer_max_new_tokens": 32}
    backend = Backend(config, device)
    adapter = NativeAdapter(backend, PROMPTS)
    return adapter
