"""Local Transformers replay. Evict complete observations while retaining user questions."""
from pathlib import Path

from PIL import Image
import torch
import transformers
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

from ..chat import ChatAdapter


class Local(ChatAdapter):
    def __init__(self, model_path, architecture, device="auto", adapter_path=None):
        self.internvl = architecture == "InternVLChatModel"
        self.size = (448, 448) if self.internvl else (308, 308)
        self.context = 40960 if self.internvl else 65536
        loading = {"torch_dtype": torch.bfloat16, "device_map": "auto" if device == "auto" else {"": device}}
        if self.internvl:
            from torchvision import transforms
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_flash_attn=True, **loading)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer.chat_template = (Path(model_path) / "chat_template.jinja").read_text()
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            self.device = self.model.language_model.get_input_embeddings().weight.device
        else:
            self.model = getattr(transformers, architecture).from_pretrained(
                model_path, attn_implementation="flash_attention_2", **loading)
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=3136, max_pixels=100352)
            self.tokenizer = self.processor.tokenizer
            self.device = self.model.get_input_embeddings().weight.device
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
        self.model.eval()
        self.generation = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True)

    def image(self, frame):
        return frame.convert("RGB").resize(self.size, Image.Resampling.BICUBIC)

    def inputs(self, messages):
        chat, images = [], []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                images.extend(content)
                content = [{"type": "image"} for _ in content]
            chat.append({"role": message["role"], "content": content})
        if self.internvl:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
            prompt = prompt.replace("<image>", "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>")
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        else:
            prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=images, add_special_tokens=False, return_tensors="pt")
        return inputs, images

    @torch.inference_mode()
    def generate(self, messages, budget):
        inputs, images = self.inputs(messages)
        while inputs.input_ids.shape[1] + budget > self.context:
            positions = [i for i, m in enumerate(messages) if isinstance(m["content"], list)]
            if len(positions) < 2:
                raise ValueError("The question and newest image exceed the model context.")
            start, end = positions[:2]
            messages[start:end] = [m for m in messages[start + 1:end] if m["role"] == "user"]
            inputs, images = self.inputs(messages)
        inputs = inputs.to(self.device)
        if self.internvl:
            pixels = torch.stack([self.transform(im) for im in images]).to(self.model.vision_model.device, torch.bfloat16)
            inputs["pixel_values"] = pixels
            inputs["visual_features"] = torch.cat([self.model.extract_feature(batch) for batch in pixels.split(4)])
            inputs["logits_to_keep"] = 1
        ids = self.model.generate(**inputs, generation_config=self.generation, max_new_tokens=budget)[0]
        ids = ids if self.internvl else ids[inputs.input_ids.shape[1]:]
        stopped = bool(len(ids)) and ids[-1].item() == self.tokenizer.eos_token_id
        return {"response": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                "protocol_error": not stopped, "hit_generation_limit": not stopped and len(ids) == budget}
