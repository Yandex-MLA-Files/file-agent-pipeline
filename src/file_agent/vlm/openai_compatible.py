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

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        timeout_seconds: float = 120,
        max_retries: int = 0,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model = model.strip()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        if not self.model:
            raise ValueError("model is required")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.vlm_describe_image") as span:
            span.set_attribute("file_agent.model", self.model)

            try:
                img_base64 = self._encode_image(image)
                request = {
                    "model": self.model,
                    "messages": [
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
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.enable_thinking is not None:
                    request["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking,
                        }
                    }

                response = self.client.chat.completions.create(**request)
                self.request_count += 1
                if not response.choices:
                    raise ValueError("VLM returned an empty response")
                description = (response.choices[0].message.content or "").strip()
                if not description:
                    raise ValueError("VLM returned an empty response")

                span.set_attribute("file_agent.response_length", len(description))
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                    self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
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
