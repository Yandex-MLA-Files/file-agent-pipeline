import base64
import io
import logging

from openai import OpenAI
from PIL import Image

from file_agent.telemetry import tracer

from .base import VLMClient

logger = logging.getLogger(__name__)


class OpenAICompatibleVLMClient(VLMClient):
    """VLM client for any OpenAI-compatible vision endpoint (Ollama, vLLM, ...)."""

    def __init__(self, base_url: str, model: str, api_key: str = "dummy", max_tokens: int = 500):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.vlm_describe_image") as span:
            span.set_attribute("file_agent.model", self.model)

            try:
                img_base64 = self._encode_image(image)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                )
                description = (response.choices[0].message.content or "").strip()

                span.set_attribute("file_agent.response_length", len(description))
                usage = getattr(response, "usage", None)
                if usage is not None:
                    span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                    span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

                return description
            except Exception as exc:
                logger.error("VLM request failed (%s): %s", self.model, exc)
                raise

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
