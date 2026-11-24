"""Frozen Qwen3-VL ViT, FP32 trainable mergers and language-model LoRA."""
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file, load_file
from model.backend import Qwen3VLBackend

from losses.cross_entropy import chunked_loss, task_mean_loss
from losses.diagnosis_attention import AttentionQKCapture, diagnosis_attention_loss


def load_base_model(model_path, device):
    backend = Qwen3VLBackend()
    base = backend.model_class.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", local_files_only=True,
        device_map={"": device},
    )
    return base, backend


class PathStreamModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        base, self.backend = load_base_model(args.model, device)
        base.config.use_cache = False
        self.backend.language_model(base).config.use_cache = False
        base.requires_grad_(False)
        if args.resume:
            self.adapter = PeftModel.from_pretrained(
                base, args.resume, is_trainable=True, torch_device=str(device),
            )
        else:
            self.adapter = get_peft_model(base, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM", target_modules=self.backend.lora_targets,
            ))
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.chunk_size = args.loss_chunk_size
        self.ce_mode = args.ce_mode
        ce = [args.losses[name] for name in ('gate_ce', 'description_ce', 'diagnosis_ce')]
        self.ce_weights = [item['weight'] for item in ce]
        self.disabled_ce = tuple(i for i, item in enumerate(ce, 1) if not item['enabled'])
        attention = args.losses['diagnosis_attention']
        self.attention_weight = attention['weight'] if attention['enabled'] else 0.0
        self.query_chunk_size = args.attention['query_chunk_size']
        self.capture = (AttentionQKCapture(self.language_model, args.attention['layer_index'])
                        if self.attention_weight else None)
        for module in self.mergers().values():
            module.float().requires_grad_(True)
        if args.resume:
            state = load_file(str(Path(args.resume) / "merger_model.safetensors"))
            for name, module in self.mergers().items():
                prefix = name + "."
                module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                        if key.startswith(prefix)})

    def mergers(self):
        return {"merger": self.vision_model.merger,
                **{f"deepstack_merger_list.{i}": module
                   for i, module in enumerate(self.vision_model.deepstack_merger_list)}}

    def train(self, mode=True):
        super().train(mode)
        self.vision_model.eval()
        for module in self.mergers().values():
            module.train(mode)
        return self

    def project_visual(self, features, deepstack):
        # Each cache row is four raw ViT patch vectors, before merger LayerNorm.
        hidden_size = self.vision_model.config.hidden_size
        with torch.autocast(device_type=features.device.type, dtype=torch.bfloat16):
            projected = self.vision_model.merger(features.reshape(-1, hidden_size))
            projected_deepstack = torch.stack([
                module(deepstack[i].reshape(-1, hidden_size))
                for i, module in enumerate(self.vision_model.deepstack_merger_list)
            ])
        return projected, projected_deepstack

    @property
    def base(self):
        return self.adapter.get_base_model()

    @property
    def language_model(self):
        return self.backend.language_model(self.base)

    @property
    def vision_model(self):
        return self.backend.vision_model(self.base)

    def rope_index(self, input_ids, image_grid_thw, attention_mask):
        return self.backend.rope_index(self.base, input_ids, image_grid_thw, attention_mask)

    def hidden_states(self, input_ids, attention_mask, visual_features=None,
                      image_grid_thw=None, deepstack_features=None, position_ids=None,
                      past_key_values=None, cache_position=None, use_cache=False):
        if visual_features is not None:
            visual_features, deepstack_features = self.project_visual(visual_features, deepstack_features)
        embeddings = self.language_model.embed_tokens(input_ids)
        image_mask = input_ids == self.base.config.image_token_id
        if visual_features is not None:
            embeddings = embeddings.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(embeddings),
                visual_features.to(embeddings.dtype),
            )
        if position_ids is None:
            position_ids = self.rope_index(input_ids, image_grid_thw, attention_mask)
        return self.backend.forward(
            self.base, visual_pos_masks=image_mask, deepstack_features=deepstack_features,
            inputs_embeds=embeddings, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, cache_position=cache_position,
            use_cache=use_cache, return_dict=True,
        )

    def forward(self, input_ids, attention_mask, visual_features, image_grid_thw,
                labels, loss_kind, attention_items, deepstack_features=None):
        if attention_items:
            self.capture.begin()
        hidden = self.hidden_states(
            input_ids, attention_mask, visual_features, image_grid_thw, deepstack_features,
        ).last_hidden_state
        state = self.capture.take() if attention_items else None
        args = (self.base.lm_head, hidden, labels, loss_kind, self.chunk_size)
        if self.ce_mode == 'token_mean':
            loss, detail = chunked_loss(*args, token_weights=self.ce_weights, disabled_kinds=self.disabled_ce)
        else:
            loss, detail = task_mean_loss(*args, task_weights=self.ce_weights, disabled_kinds=self.disabled_ce)
        kl, attention_metrics = diagnosis_attention_loss(state, attention_items, self.query_chunk_size,
                                                        loss.detach().new_zeros(()))
        return loss + self.attention_weight * kl, detail, attention_metrics

    def save(self, directory, processor):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(self.adapter, save_embedding_layers=False)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in state.items()},
            str(directory / "adapter_model.safetensors"),
        )
        merger_state = {name + "." + key: value.detach().cpu().contiguous()
                        for name, module in self.mergers().items()
                        for key, value in module.state_dict().items()}
        save_file(merger_state, str(directory / "merger_model.safetensors"))
        self.adapter.peft_config["default"].save_pretrained(directory)
        processor.save_pretrained(directory)
