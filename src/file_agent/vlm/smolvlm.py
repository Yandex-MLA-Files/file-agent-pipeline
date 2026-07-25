import logging

from PIL import Image

from .base import VLMClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


class SmolVLMClient(VLMClient):
    """Local VLM built on SmolVLM — the cheapest way to describe figures.

    SmolVLM-256M is a ~500 MB vision-language model that runs on CPU with no
    server, no API key and no per-image cost, which makes it the default choice
    for describing a handful of figures per document. Quality is below large
    hosted VLMs, so an OpenAI-compatible endpoint remains available for setups
    that need better captions (see :mod:`file_agent.vlm.factory`).

    The model is loaded lazily on the first call so importing this module (or
    building a client that is never used) costs nothing.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 200):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # local import: torch is heavy and already a docling dependency
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading local VLM %s (first use only)...", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, dtype=torch.float32
        )
        self._model.eval()

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        import torch

        self._load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=chat_prompt, images=[image.convert("RGB")], return_tensors="pt"
        )

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
