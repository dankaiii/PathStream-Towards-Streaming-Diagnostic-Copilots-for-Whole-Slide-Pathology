"""Hosted models share image replay and use their respective API message formats."""
import base64
import io

from PIL import Image
from ..chat import ChatAdapter


class Hosted(ChatAdapter):
    def __init__(self, client, model, *, responses):
        self.client, self.model, self.responses = client, model, responses

    def image(self, frame):
        buffer = io.BytesIO()
        frame.convert("RGB").resize((308, 308), Image.Resampling.BICUBIC).save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        if self.responses:
            return {"type": "input_image", "image_url": url, "detail": "auto"}
        return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}

    def generate(self, messages, budget):
        if self.responses:
            result = self.client.responses.create(model=self.model, input=messages, store=False,
                max_output_tokens=budget, reasoning={"effort": "none"})
            return {"response": result.output_text.strip(), "protocol_error": result.status != "completed",
                    "hit_generation_limit": result.status == "incomplete"}
        result = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
            max_tokens=budget, extra_body={"enable_thinking": False})
        choice = result.choices[0]
        return {"response": choice.message.content.strip(), "protocol_error": choice.finish_reason != "stop",
                "hit_generation_limit": choice.finish_reason == "length"}
