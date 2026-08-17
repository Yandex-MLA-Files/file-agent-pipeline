import base64
import io
import logging
from typing import Any

from openai import OpenAI
from PIL import Image

from file_agent.telemetry import tracer

from .base import VLMClient

logger = logging.getLogger(__name__)

# Images larger than this (longest side, pixels) are downscaled before upload:
# vision encoders tile the input, so a 4000-px scan costs many times the tokens
# of a 1600-px one for no gain in legibility.
MAX_IMAGE_SIDE = 1600


class OpenAICompatibleVLMClient(VLMClient):
    """VLM client for any OpenAI-compatible vision endpoint (vLLM, Ollama, ...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        max_tokens: int = 500,
        timeout: float | None = None,
        enable_thinking: bool | None = None,
        temperature: float = 0.0,
    ):
        client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
            client_kwargs["max_retries"] = 1
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.temperature = temperature

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        return self.describe_image_verbose(image, prompt, max_tokens=max_tokens)[0]

    def describe_image_verbose(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int | None = None,
        max_image_side: int | None = None,
    ) -> tuple[str, str | None]:
        with tracer.start_as_current_span("file_agent.vlm_describe_image") as span:
            span.set_attribute("file_agent.model", self.model)

            try:
                img_base64 = self._encode_image(image, max_image_side)
                extra: dict[str, Any] = {}
                if self.enable_thinking is not None:
                    extra["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                    }
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
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                    **extra,
                )
                choice = response.choices[0]
                description = (choice.message.content or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)

                span.set_attribute("file_agent.response_length", len(description))
                if finish_reason:
                    span.set_attribute("file_agent.finish_reason", str(finish_reason))
                usage = getattr(response, "usage", None)
                if usage is not None:
                    span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                    span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

                return description, finish_reason
            except Exception as exc:
                logger.error("VLM request failed (%s): %s", self.model, exc)
                raise

    @staticmethod
    def _encode_image(image: Image.Image, max_side: int | None = None) -> str:
        image = image.convert("RGB")
        limit = max_side or MAX_IMAGE_SIDE
        longest = max(image.width, image.height)
        if longest > limit:
            scale = limit / longest
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
