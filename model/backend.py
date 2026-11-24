"""Qwen3-VL image features, DeepStack injection and language-only LoRA targets."""
import torch
from transformers import Qwen3VLForConditionalGeneration


LANGUAGE_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)


class Qwen3VLBackend:
    model_class = Qwen3VLForConditionalGeneration
    lora_targets = LANGUAGE_LORA_TARGETS

    def language_model(self, base):
        return base.model.language_model

    def vision_model(self, base):
        return base.model.visual

    def rope_index(self, base, input_ids, image_grid_thw, attention_mask):
        return base.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask,
        )[0]

    def encode_images(self, base, pixels, grid):
        features, deepstack = base.model.visual(pixels, grid_thw=grid)
        return {"features": features, "deepstack_features": torch.stack(deepstack)}

    def forward(self, base, visual_pos_masks, deepstack_features, **kwargs):
        if kwargs["cache_position"] is not None:
            # Cached chunks need linear attention positions separate from 3-axis visual RoPE.
            positions = kwargs["position_ids"]
            text_positions = kwargs["cache_position"].view(1, 1, -1).expand(
                1, positions.shape[1], -1,
            )
            kwargs["position_ids"] = torch.cat((text_positions, positions), dim=0)
        deepstack = None if deepstack_features is None else list(deepstack_features.unbind(0))
        return base.model.language_model(
            **kwargs, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack,
        )
