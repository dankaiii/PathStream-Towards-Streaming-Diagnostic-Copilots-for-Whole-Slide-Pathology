"""LiveCC domain-SFT backbone and streaming message format."""
import numpy as np
from transformers import Qwen2VLForConditionalGeneration


SILENCE = " ..."
SILENCE_INSTRUCTION = "Pause"
DESCRIPTION_SUFFIX = SILENCE


def training_model_class():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

    apply_liger_kernel_to_qwen2_vl()
    return Qwen2VLForConditionalGeneration


def frame_content(index, fps):
    return [{"type": "text", "text": f"Time={index / fps:.1f}-{(index + 1) / fps:.1f}s"},
            {"type": "video"}]


def encode(processor, messages, frames, max_pixels):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    videos = [np.array(frame)[None] for frame in frames]
    return processor(text=text, videos=videos, return_tensors="pt", videos_kwargs={
        "do_sample_frames": False, "size": {"shortest_edge": 3136, "longest_edge": max_pixels},
    })
