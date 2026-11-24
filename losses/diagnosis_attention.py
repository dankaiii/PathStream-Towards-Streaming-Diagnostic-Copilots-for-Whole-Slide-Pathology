# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable, causal evidence attention for multimodal streaming SFT.

Adapted from Nrich-sunny/Attend-to-Evidence (Apache-2.0):
verl/workers/actor/trainable_attention.py and dp_actor.py.

Uses one decoder layer, head-mean model-to-target KL and background smoothing.
Unlike the upstream single-image implementation, keys can be multiple image spans
or historical caption spans. The backbone keeps ordinary FlashAttention.
"""
import torch
from torch.utils.checkpoint import checkpoint

ATTENTION_GROUPS = ('diagnosis_visual', 'diagnosis_text')


class AttentionQKCapture:
    """Capture actual post-QK-norm projections, before RoPE, with their gradients."""
    def __init__(self, language_model, layer_index):
        index = (2 * len(language_model.layers)) // 3 if layer_index < 0 else layer_index
        self.layer_index = index
        self.module = language_model.layers[index].self_attn
        self.active = False
        self.q = self.k = self.state = None
        q_source, k_source = self.module.q_norm, self.module.k_norm
        self.handles = [q_source.register_forward_hook(self.capture_q),
                        k_source.register_forward_hook(self.capture_k),
                        self.module.register_forward_hook(self.capture_state, with_kwargs=True)]

    def capture_q(self, module, inputs, output):
        if self.active:
            self.q = output

    def capture_k(self, module, inputs, output):
        if self.active:
            self.k = output

    def capture_state(self, module, args, kwargs, output):
        if not self.active:
            return
        q, k = self.q, self.k
        cos, sin = kwargs['position_embeddings']
        self.state = (q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        self.q = self.k = None

    def begin(self):
        self.state = None
        self.active = True

    def take(self):
        # Checkpoint backward recomputations must not retain another forward graph.
        self.active = False
        state, self.state = self.state, None
        return state


def rotate(tensor, cos, sin):
    half = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
    return tensor * cos + rotated * sin


def head_mean_kl(query, key, target, scale):
    """Normalize independently within one modality, then average actual heads."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        heads, queries, width = query.shape
        key_heads = key.shape[0]
        # GQA's adjacent query heads share a key head. Batch them without
        # materializing repeated K tensors; retain FP32 scores and head-mean KL.
        grouped_query = query.reshape(key_heads, -1, width).float()
        scores = torch.bmm(grouped_query, key.float().transpose(1, 2)) * scale
        scores = scores.reshape(heads, queries, key.shape[1])
        probabilities = scores.softmax(dim=-1).mean(dim=0)
        # float32 softmax may underflow; keep log(0) out of the KL calculation.
        log_probability = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        return (probabilities * (log_probability - target.log().unsqueeze(0))).sum(-1).mean()


def attention_slice_kl(query, key, cos, sin, rows, columns, target, scale):
    # Slice inside checkpoint: do not retain a separate history K copy per frame.
    q = query[0].index_select(1, rows)
    k = key[0].index_select(1, columns)
    q = rotate(q, cos[0].index_select(0, rows).unsqueeze(0), sin[0].index_select(0, rows).unsqueeze(0))
    k = rotate(k, cos[0].index_select(0, columns).unsqueeze(0), sin[0].index_select(0, columns).unsqueeze(0))
    return head_mean_kl(q, k, target, scale)


def diagnosis_attention_loss(state, items, chunk_size, zero):
    """Average KL within each available modality, then across modalities."""
    values = [[], []]
    if items:
        query, key, cos, sin = state
        scale = query.shape[-1] ** -0.5
        for item in items:
            rows = item['queries']
            chunks = []
            for chunk in rows.split(chunk_size):
                kl = checkpoint(attention_slice_kl, query, key, cos, sin, chunk,
                                item['keys'], item['target'], scale, use_reentrant=False)
                chunks.append(kl * (chunk.numel() / rows.numel()))
            values[item['kind']].append(torch.stack(chunks).sum())
    means = [torch.stack(group).mean() for group in values if group]
    loss = torch.stack(means).mean() if means else zero
    sums = [torch.stack(group).sum().detach() if group else zero.detach() for group in values]
    counts = zero.new_tensor([len(group) for group in values])
    return loss, torch.cat((torch.stack(sums), counts))
