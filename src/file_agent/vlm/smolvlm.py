import logging

from PIL import Image

from .base import VLMClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


class SmolVLMClient(VLMClient):
    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 200):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # local import: torch is heavy and already a docling dependency
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading local VLM %s (first use only)...", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, dtype=torch.float16
                ).to("cuda")
                self._device = "cuda"
            except RuntimeError as exc:
                logger.warning("VLM failed to load on GPU (%s); falling back to CPU.", exc)
                torch.cuda.empty_cache()
                self._model = None

        if self._model is None:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, dtype=torch.float32
            )
            self._device = "cpu"

        logger.info("VLM %s loaded on %s", self.model_name, self._device)
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
        ).to(self._device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
