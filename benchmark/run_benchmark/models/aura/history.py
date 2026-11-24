"""AURA context management, adapted from its official ContextManaged inference code."""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter, deque


SILENT_TEXT = "<|silent|>"


_PENALTY_PUNCT = frozenset(
    ".,!?;:，。！？；：、'\"()[]{}""''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》"
)


class SessionHistory:
    def __init__(self, max_rounds: int = 20, num_rounds_keep: int = 15,
                 pruning_enabled: bool = False, debug_context_file: str = None,
                 max_context_qas: int = 10, max_1qna_rounds: int = 4):
        self.history = []
        self.max_rounds = max_rounds
        self.num_rounds_keep = num_rounds_keep
        self.pruning_enabled = pruning_enabled
        self.current_rounds = 0
        self.debug_context_file = debug_context_file
        self.max_context_qas = max_context_qas
        self.max_1qna_rounds = max_1qna_rounds

        self.system_prompt = "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user's message or the visual context. Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
        self._system_msg = {"role": "system", "content": self.system_prompt}
        self._context_history = []   # list of QAs; each QA = list of message dicts (text-only)
        self._sliding_window = []    # list of message dicts (may contain multimedia)
        self._reset()

    def _reset(self):
        """Reset history to initial state (complete reset)."""
        self._context_history = []
        self._sliding_window = []
        self._rebuild_history()
        self.current_rounds = 0

    def _rebuild_history(self):
        """Compose self.history = [system] + context_history msgs + sliding_window msgs."""
        self.history = [self._system_msg]
        for qa in self._context_history:
            self.history.extend(qa)
        self.history.extend(self._sliding_window)

    def _sw_round_count(self):
        """Count user messages (rounds) in the sliding window."""
        return sum(1 for m in self._sliding_window if m["role"] == "user")

    def _extract_user_text(self, content) -> str:
        """
        从 user message 的 content 中提取纯文字。

        Args:
            content: 可能是 str 或 list[dict]

        Returns:
            字符串（可能为空字符串 ""）
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts)  # 可能为空字符串 ""

        return ""

    def _is_silent_response(self, content) -> bool:
        """检查 assistant 回复是否为 <|silent|>"""
        if isinstance(content, str):
            return content.strip() == SILENT_TEXT
        return False

    def _has_user_text(self, user_msg) -> bool:
        """检查 user 消息是否包含实际文本内容（非空）。"""
        content = user_msg.get("content", [])
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text", "").strip():
                    return True
        return False

    def _parse_sw_rounds(self):
        """Parse sliding window messages into (user_msg, assistant_msg|None) pairs."""
        rounds = []
        i = 0
        while i < len(self._sliding_window):
            msg = self._sliding_window[i]
            if msg["role"] == "user":
                user_msg = msg
                assistant_msg = None
                if (i + 1 < len(self._sliding_window) and
                        self._sliding_window[i + 1]["role"] == "assistant"):
                    assistant_msg = self._sliding_window[i + 1]
                    i += 2
                else:
                    i += 1
                rounds.append((user_msg, assistant_msg))
            else:
                i += 1
        return rounds

    def _group_rounds_into_qas(self, rounds):
        """
        Group rounds into QA units for context management.
        A round whose user message contains text starts a new QA;
        subsequent video-only rounds are continuations of the same QA.
        """
        groups = []
        current_group = []
        for user_msg, assistant_msg in rounds:
            if self._has_user_text(user_msg):
                if current_group:
                    groups.append(current_group)
                current_group = [(user_msg, assistant_msg)]
            else:
                current_group.append((user_msg, assistant_msg))
        if current_group:
            groups.append(current_group)
        return groups

    def _rewrite_qa_for_history(self, qa_rounds):
        """
        Apply rewrite rules A & B to a QA group from the sliding window.
        Rule A: Remove <video>, keep only text in user messages.
        Rule B: Remove entire round if assistant is <|silent|>.
        Returns a list of message dicts (context history format), or None.
        """
        rewritten = []
        for user_msg, assistant_msg in qa_rounds:
            if assistant_msg and self._is_silent_response(assistant_msg["content"]):
                continue
            user_text = self._extract_user_text(user_msg["content"])
            rewritten.append({"role": "user", "content": user_text})
            if assistant_msg:
                rewritten.append({"role": "assistant", "content": assistant_msg["content"]})
        return rewritten if rewritten else None

    def _qa_to_round_pairs(self, qa_messages):
        """Convert flat QA message list → [(user_content, assistant_content), ...]."""
        pairs = []
        i = 0
        while i < len(qa_messages):
            if qa_messages[i]["role"] == "user":
                u = qa_messages[i]["content"]
                a = None
                if i + 1 < len(qa_messages) and qa_messages[i + 1]["role"] == "assistant":
                    a = qa_messages[i + 1]["content"]
                    i += 2
                else:
                    i += 1
                pairs.append((u, a))
            else:
                i += 1
        return pairs

    def _count_qa_rounds(self, qa_messages):
        """Count rounds (user messages) in a QA."""
        return sum(1 for m in qa_messages if m["role"] == "user")

    def _classify_qa(self, qa_messages):
        """
        Classify a context-history QA (§2.2).
        Returns: "basic" | "1q1a" | "1qna" | "truncated" | None
        """
        pairs = self._qa_to_round_pairs(qa_messages)
        n = len(pairs)
        if n == 0:
            return None
        first_has_text = bool(pairs[0][0] and pairs[0][0].strip())
        if n == 1:
            return "basic" if first_has_text else "truncated"
        if n == 2 and first_has_text:
            return "1q1a"
        if n >= 3 and first_has_text:
            return "1qna"
        if not first_has_text:
            return "truncated"
        return None

    def _enforce_1qna_limit(self, qa_messages):
        """
        Rule E: 1QNA total rounds ≤ max_1qna_rounds (default 4).
        Delete earliest "" + assistant round (skipping the first round).
        """
        while self._count_qa_rounds(qa_messages) > self.max_1qna_rounds:
            found = False
            i = 2  # never delete the first round (indices 0, 1)
            while i < len(qa_messages):
                if (qa_messages[i]["role"] == "user"
                        and qa_messages[i]["content"] == ""):
                    del qa_messages[i]
                    if i < len(qa_messages) and qa_messages[i]["role"] == "assistant":
                        del qa_messages[i]
                    found = True
                    break
                i += 1
            if not found:
                break

    def _merge_truncated_qa(self, truncated_messages):
        """
        Rule D: Merge a truncated QA ("" + assistant) with the last QA in
        context history.  After merge, enforce Rule E if it became 1QNA.
        """
        if self._context_history:
            last_qa = self._context_history[-1]
            last_qa.extend(truncated_messages)
            if self._count_qa_rounds(last_qa) > self.max_1qna_rounds:
                self._enforce_1qna_limit(last_qa)
        else:
            self._context_history.append(list(truncated_messages))

    def _prune_history(self):
        """
        Context management per context.md §3:

        §3.1 — Trigger & migration:
            When sliding window rounds > max_rounds, keep the last
            num_rounds_keep rounds in sliding window; move the earlier
            (total - num_rounds_keep) rounds to context history (hard cut).

        §3.2 — Rewrite rules applied to moved rounds:
            A: Remove <video> from user messages
            B: Remove silent rounds
            C: Validate each QA matches §2.2 types
            D: Merge truncated QAs with last context history QA
            E: 1QNA in context history ≤ max_1qna_rounds rounds

        §3.3 — Context history capacity ≤ max_context_qas QAs.
        """
        rounds = self._parse_sw_rounds()
        if len(rounds) <= self.max_rounds:
            return

        num_to_move = len(rounds) - self.num_rounds_keep
        if num_to_move <= 0:
            return
        rounds_to_move = rounds[:num_to_move]
        rounds_remaining = rounds[num_to_move:]

        qa_groups = self._group_rounds_into_qas(rounds_to_move)

        for qa_rounds in qa_groups:
            head_has_text = self._has_user_text(qa_rounds[0][0])
            rewritten = self._rewrite_qa_for_history(qa_rounds)
            if not rewritten:
                continue

            if head_has_text:
                qa_type = self._classify_qa(rewritten)
                if qa_type == "1qna":
                    self._enforce_1qna_limit(rewritten)
                self._context_history.append(rewritten)
            else:
                # Orphan continuation rounds → split into individual truncated
                # QAs and merge each via Rule D.
                i = 0
                while i < len(rewritten):
                    if rewritten[i]["role"] == "user":
                        truncated = [rewritten[i]]
                        if (i + 1 < len(rewritten)
                                and rewritten[i + 1]["role"] == "assistant"):
                            truncated.append(rewritten[i + 1])
                            i += 2
                        else:
                            i += 1
                        self._merge_truncated_qa(truncated)
                    else:
                        i += 1

        # §3.3: enforce capacity limit
        while len(self._context_history) > self.max_context_qas:
            self._context_history.pop(0)

        # Rebuild sliding window from remaining rounds
        self._sliding_window = []
        for user_msg, assistant_msg in rounds_remaining:
            self._sliding_window.append(user_msg)
            if assistant_msg is not None:
                self._sliding_window.append(assistant_msg)

        self.current_rounds = self._sw_round_count()
        self._rebuild_history()

    def add_user_message(self, text: str, images: list = None, video_tuple: tuple = None):
        """
        Add user message with optional images or video.

        Args:
            text: User text message
            images: List of PIL images (for image mode)
            video_tuple: Tuple of (numpy_array, metadata_dict) for video mode
                        numpy_array shape: (num_frames, height, width, 3)
                        metadata_dict: {"fps": float, "duration": float, "total_num_frames": int, ...}
        """
        content = []

        if video_tuple:
            content.append({"type": "video", "video": video_tuple})
        elif images:
            content.extend([{"type": "image", "image": img} for img in images])

        if text:
            content.append({"type": "text", "text": text})
        elif not images and not video_tuple:
            return

        msg = {"role": "user", "content": content}
        self._sliding_window.append(msg)
        self.history.append(msg)
        self.current_rounds += 1

        # §3.1: Trigger pruning when sliding window rounds exceed max_rounds
        if self.pruning_enabled and self._sw_round_count() > self.max_rounds:
            self._prune_history()

    def add_assistant_message(self, text: str):
        """Add assistant response to history."""
        msg = {"role": "assistant", "content": text}
        self._sliding_window.append(msg)
        self.history.append(msg)

    def get_vllm_inputs(self):
        """
        Construct prompt and multi_modal_data for vLLM.
        MUST include ALL history media to match the text prompt for Prefix Caching.
        Supports both images and videos.
        """
        full_prompt = ""
        all_images = []
        all_videos = []

        for msg in self.history:
            role = msg["role"]
            content = msg["content"]

            full_prompt += f"<|im_start|>{role}"

            if isinstance(content, str):
                full_prompt += content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        full_prompt += item.get("text", "")
                    elif item.get("type") == "image":
                        # Qwen3-VL image tokens
                        full_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
                        all_images.append(item.get("image"))
                    elif item.get("type") == "video":
                        # Qwen3-VL video tokens
                        full_prompt += "<|vision_start|><|video_pad|><|vision_end|>"
                        all_videos.append(item.get("video"))

            full_prompt += "<|im_end|>"

        # Add generation prompt
        full_prompt += "<|im_start|>assistant"

        # Build multi_modal_data
        multi_modal_data = {}
        if all_images:
            multi_modal_data["image"] = all_images
        if all_videos:
            multi_modal_data["video"] = all_videos

        return {
            "prompt": full_prompt,
            "multi_modal_data": multi_modal_data
        }


class CrossTurnPenalty:
    """AURA soft logit bias over the ten most recent turns (no n-gram blocking)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.history = deque(maxlen=10)

    def record(self, response_text=None):
        self.history.append(response_text)

    def build_sampling_kwargs(self):
        spoken = [Counter(self.tokenizer.encode(text, add_special_tokens=False))
                  for text in self.history if text]
        presence = Counter(token for counts in spoken for token in counts)
        bias = Counter()
        for i, counts in enumerate(spoken, 1):
            for token, count in counts.items():
                if presence[token] < 2 or token in self.tokenizer.all_special_ids:
                    continue
                word = self.tokenizer.decode([token]).strip()
                if word and not word.isdigit() and not all(c in _PENALTY_PUNCT for c in word):
                    bias[token] += min(count, 3) * i / len(spoken)
        return {"logit_bias": {token: -min(value, 100) for token, value in bias.most_common(500)}} if bias else {}
