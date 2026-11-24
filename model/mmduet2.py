"""MM-Duet2 domain-SFT backbone and streaming message format."""
from transformers import Qwen2_5_VLForConditionalGeneration


SILENCE = "NO REPLY"
SILENCE_INSTRUCTION = "Output NO REPLY"
DESCRIPTION_SUFFIX = ""


def training_model_class():
    return Qwen2_5_VLForConditionalGeneration


def frame_content(index, fps):
    return [{"type": "image"}]


def encode(processor, messages, frames, max_pixels):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # The training template ends with the final assistant EOS.
    return processor(text=text.removesuffix("\n"), images=frames, return_tensors="pt",
                     images_kwargs={"min_pixels": 3136, "max_pixels": max_pixels})
