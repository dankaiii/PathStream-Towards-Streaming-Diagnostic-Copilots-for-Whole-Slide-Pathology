"""Sequential chat replay for hosted and direct Transformers models."""
from .base import Adapter

INSTRUCTION = (
    "Observe the pathology views in their presented order. For each frame, output "
    "either exactly the lowercase word silent, or directly a concise English "
    "description of the complete morphology of that view. Describe the first frame. "
    "Output silent only while a similar view continues after your description. "
    "A finding may be described again in another view; do not restrict descriptions "
    "to facts absent from the entire history. Do not add an update label, heading, "
    "reasoning, or a diagnosis. Do not give a final diagnosis."
)
SYSTEM = "You are an expert pathologist. Observe the image sequence in order and answer the user's question."
PROTOCOL = (
    "For each pathology frame, output either exactly the lowercase word silent, "
    "or directly a concise English morphological description of that frame. "
    "Describe the first frame. Output silent only while a similar view continues "
    "after your description. A finding may be described again in another view. "
    "Do not add an update label, heading, reasoning, or a diagnosis to a frame description. "
    "When the final diagnostic question arrives, answer it directly using all observed views; "
    "the silent option applies only to frame observations."
)

class ChatAdapter(Adapter):
    def start(self, task, system, instruction):
        if task == 1:
            instruction += " If remaining silent, output exactly silent. Otherwise output the description directly, without an update label."
        elif task == 2:
            instruction = INSTRUCTION
        else:
            system += "\n\n" + PROTOCOL
        return ChatSession(self, system, instruction)

    def answer_frames(self, frames, prompt, budget):
        messages = [{"role": "system", "content": SYSTEM}]
        messages.extend({"role": "user", "content": [self.image(frame)]} for frame in frames)
        messages.append({"role": "user", "content": prompt})
        return self.generate(messages, budget)


class ChatSession:
    def __init__(self, model, system, instruction):
        self.model, self.instruction = model, instruction
        self.messages = [{"role": "system", "content": system}]

    def observe(self, frame, index, fps, budget):
        self.messages.append({"role": "user", "content": [self.model.image(frame)]})
        if index == 0:
            self.messages.append({"role": "user", "content": self.instruction})
        result = self.model.generate(self.messages, budget)
        raw = result["response"].strip()
        silent = raw.casefold().rstrip(".。!！") == "silent"
        bad = result["protocol_error"] or not raw
        if not bad and not silent:
            self.messages.append({"role": "assistant", "content": raw})
        return {**result, "raw_response": raw, "response": "" if silent and not bad else raw,
                "protocol_error": bad, "decision": "protocol_error" if bad else "silent" if silent else "update"}

    def diagnose(self, question, budget):
        self.messages.append({"role": "user", "content": question})
        result = self.model.generate(self.messages, budget)
        text = result["response"].strip()
        bad = result["protocol_error"] or not text or text.casefold().rstrip(".。!！") == "silent"
        return {**result, "protocol_error": bad, "decision": "protocol_error" if bad else "answer"}
