"""Apply language LoRA and train the visual merger for native domain SFT."""
from copy import copy

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoProcessor

from model.backend import LANGUAGE_LORA_TARGETS


def load_model(protocol, path, config):
    model_class = protocol.training_model_class()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
    base = model_class.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                       device_map={"": torch.cuda.current_device()}, local_files_only=True)
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LANGUAGE_LORA_TARGETS,
        modules_to_save=["merger"],
    ))
    base.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.model.visual.merger.modules_to_save["default"].float()
    set_training_mode(model)
    return model, processor


def set_training_mode(model):
    model.train()
    visual = model.get_base_model().model.visual
    visual.eval()
    visual.merger.modules_to_save["default"].train()


def save_adapter(model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    save_file({name: value.detach().cpu().contiguous() for name, value in state.items()},
              str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})
    config = copy(model.peft_config["default"])
    config.inference_mode = True
    config.save_pretrained(directory)
