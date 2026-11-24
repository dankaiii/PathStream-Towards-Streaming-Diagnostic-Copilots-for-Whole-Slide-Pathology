"""Bounded streaming history; retain instructions and evict complete oldest turns."""
from dataclasses import dataclass, field

import torch

from data.conversation import (
    DEFAULT_MAX_CONTEXT_TOKENS, EOT, START, assistant_prefix, encode, message_tokens, turns_to_evict,
)


def context_config(max_context_tokens):
    return {
        "max_context_tokens": max_context_tokens,
        "history": "windowed_model_generated_history_no_reference_answers",
        "context_policy": "keep_instructions_evict_oldest_complete_turns_rebuild_kv_no_silent_eos",
    }


@dataclass
class HistoryTurn:
    chunks: list = field(default_factory=list)
    instructions: list = field(default_factory=list)
    removable_tokens: int = 0


class StreamingSession:
    def __init__(self, model, tokenizer, device, eos_threshold=0.0,
                 max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS):
        self.model = model
        self.base = model.base
        self.tokenizer = tokenizer
        self.device = device
        self.eos_threshold = eos_threshold
        self.text_temperature = 0.0
        self.max_context_tokens = max_context_tokens
        self.eos_id = tokenizer.convert_tokens_to_ids(EOT)
        self.start_id = tokenizer.convert_tokens_to_ids(START)
        self.past_key_values = None
        self.token_count = 0
        self.next_position = 0
        self.last_hidden = None
        self.history = [HistoryTurn()]  # System and instructions from evicted turns.
        self.context_evictions = 0
        self.evicted_turns = 0
        self.evicted_tokens = 0

    def context_stats(self):
        return {
            "context_tokens_after_turn": self.token_count,
            "context_evictions": self.context_evictions,
            "evicted_turns": self.evicted_turns,
            "evicted_tokens": self.evicted_tokens,
        }

    def remember(self, turn, tokens, features=None, grid=None, deepstack_features=None, *, instruction=False):
        if self.max_context_tokens is None:
            return
        # Keep only the retained window on CPU, including Qwen3 DeepStack features.
        tensors = tuple(value.detach().to(device="cpu", copy=True) if value is not None else None
                        for value in (features, grid, deepstack_features))
        chunk = (list(tokens), *tensors)
        turn.chunks.append(chunk)
        if instruction:
            turn.instructions.append(chunk)
        else:
            turn.removable_tokens += len(tokens)

    def reserve(self, required_tokens, *, new_frame=False):
        if self.max_context_tokens is None or self.token_count + required_tokens <= self.max_context_tokens:
            return
        # A new frame can replace all previous frames. A reply keeps its own observation.
        candidates = self.history[1:] if new_frame else self.history[1:-1]
        count = turns_to_evict(self.token_count + required_tokens - self.max_context_tokens,
                               (turn.removable_tokens for turn in candidates), self.max_context_tokens)
        for turn in candidates[:count]:
            # Move only the user text, in order, when its original observation is evicted.
            self.history[0].chunks.extend(turn.instructions)
            self.evicted_tokens += turn.removable_tokens
        del self.history[1:1 + count]
        self.evicted_turns += count
        self.context_evictions += 1
        # Rebuild only after eviction: cached keys already contain their old RoPE positions.
        self.past_key_values = None
        self.last_hidden = None
        self.token_count = self.next_position = 0
        for turn in self.history:
            for tokens, *tensors in turn.chunks:
                self._forward_tokens(tokens, *(value.to(self.device) if value is not None else None
                                               for value in tensors))

    def generation_budget(self, max_new_tokens):
        return len(assistant_prefix(self.tokenizer)) + max_new_tokens + len(encode(self.tokenizer, "\n"))

    def append_tokens(self, tokens, features=None, grid=None, deepstack_features=None):
        self._forward_tokens(tokens, features, grid, deepstack_features)

    @torch.inference_mode()
    def _forward_tokens(self, tokens, features=None, grid=None, deepstack_features=None):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if grid is not None:
            positions = self.model.rope_index(ids, grid, torch.ones_like(ids))
            positions += self.next_position
            self.next_position = int(positions.max().item()) + 1
        else:
            positions = torch.arange(
                self.next_position, self.next_position + len(tokens), device=self.device,
            ).view(1, 1, -1).expand(3, 1, -1)
            self.next_position += len(tokens)
        cache_positions = torch.arange(
            self.token_count, self.token_count + len(tokens), device=self.device,
        )
        self.token_count += len(tokens)
        output = self.model.hidden_states(
            input_ids=ids, visual_features=features, image_grid_thw=grid,
            deepstack_features=deepstack_features,
            attention_mask=torch.ones((1, self.token_count), dtype=torch.long, device=self.device),
            position_ids=positions, cache_position=cache_positions,
            past_key_values=self.past_key_values, use_cache=True,
        )
        self.past_key_values = output.past_key_values
        self.last_hidden = output.last_hidden_state[:, -1:]

    def append_message(self, message, features=None, grid=None, deepstack_features=None,
                       max_new_tokens=0):
        count = int(grid.prod(-1).item()) // 4 if grid is not None else None
        tokens, _, _ = message_tokens(message, self.tokenizer, count)
        budget = self.generation_budget(max_new_tokens) if max_new_tokens else 0
        self.reserve(len(tokens) + budget, new_frame=grid is not None)
        if message["role"] == "system":
            turn = self.history[0]
        else:
            if grid is not None or len(self.history) == 1:
                self.history.append(HistoryTurn())
            turn = self.history[-1]
        self.append_tokens(tokens, features, grid, deepstack_features)
        self.remember(turn, tokens, features, grid, deepstack_features,
                      instruction=message["role"] == "user" and grid is None)

    @torch.inference_mode()
    def next_token(self, eos_threshold=0.0):
        logits = self.base.lm_head(self.last_hidden)[0, 0].float()
        eos_probability = logits.softmax(-1)[self.eos_id].item()
        # Apply only at frame decisions; retain the original full-vocabulary probability.
        if eos_probability < eos_threshold:
            logits[self.eos_id] = -torch.inf
        return int(logits.argmax().item()), eos_probability

    def text_token(self):
        logits = self.base.lm_head(self.last_hidden)[0, 0]
        if not self.text_temperature:
            return int(logits.argmax().item())
        values, indices = logits.float().topk(32)
        probabilities = (values / self.text_temperature).softmax(-1)
        return int(indices[torch.multinomial(probabilities, 1)].item())

    @torch.inference_mode()
    def generate(self, max_new_tokens):
        self.reserve(self.generation_budget(max_new_tokens))
        input_tokens = self.token_count
        prefix = assistant_prefix(self.tokenizer)
        self.append_tokens(prefix)
        first_token = self.text_token()
        generated = []
        token = first_token
        stopped = False
        for _ in range(max_new_tokens):
            generated.append(token)
            self.append_tokens([token])
            if token == self.eos_id:
                stopped = True
                break
            token = self.text_token()
        suffix = encode(self.tokenizer, "\n") if stopped else []
        if suffix:
            self.append_tokens(suffix)
        self.remember(self.history[-1], prefix + generated + suffix)
        body = generated[:-1] if stopped else generated
        return {
            "response": self.tokenizer.decode(
                body, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            ).strip(),
            "input_tokens": input_tokens, "generated_tokens": len(generated),
            "first_token_id": first_token, "first_body_token_id": first_token,
            "assistant_header_valid": True, "assistant_header_source": "provided",
            "assistant_prefix_tokens": len(prefix),
            "stopped_on_eos": stopped, "hit_generation_limit": not stopped,
            "protocol_error": not stopped,
            **self.context_stats(),
        }

    def observe(self, message, features, grid, max_new_tokens, deepstack_features=None):
        """At a frame gate, <|im_start|> opens a reply and <|im_end|> signals silence."""
        self.append_message(message, features, grid, deepstack_features, max_new_tokens=max_new_tokens)
        token, eos_probability = self.next_token(self.eos_threshold)
        if token != self.start_id:
            silent = token == self.eos_id
            # Keep an invalid gate token in the trace. The runner stops this sample.
            return {
                "decision": "silent" if silent else "protocol_error",
                "response": "", "raw_response": "" if silent else self.tokenizer.decode([token]),
                "first_token_id": token, "gate_token_id": token,
                "eos_probability_full_vocabulary": eos_probability,
                "input_tokens": self.token_count, "generated_tokens": 0 if silent else 1,
                "assistant_header_valid": None if silent else False,
                "stopped_on_eos": False, "hit_generation_limit": False,
                "protocol_error": not silent,
                **self.context_stats(),
            }
        # The predicted start token has not entered the cache; append the full prefix once.
        result = self.generate(max_new_tokens)
        return {
            **result, "decision": "protocol_error" if result["protocol_error"] else "update",
            "first_token_id": token, "gate_token_id": token,
            "eos_probability_full_vocabulary": eos_probability,
        }

    def answer(self, question, max_new_tokens):
        self.append_message(question, max_new_tokens=max_new_tokens)
        # Explicit queries always receive an assistant prefix and bypass the stream gate.
        result = self.generate(max_new_tokens)
        return {**result, "decision": "protocol_error" if result["protocol_error"] else "answer",
                "gate_token_id": None, "eos_probability_full_vocabulary": None}
