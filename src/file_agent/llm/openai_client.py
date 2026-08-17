import logging
from typing import Any

from openai import OpenAI

from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


class OpenAILLMClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        enable_thinking: bool | None = None,
    ) -> None:
        self.client = client
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Reasoning models (Qwen3 family) think by default when served by vLLM
        # with a reasoning parser. ``None`` keeps the server default; ``True`` /
        # ``False`` are forwarded through ``chat_template_kwargs`` so the same
        # client works both with and without a reasoning parser.
        self.enable_thinking = enable_thinking

        if not self.model:
            raise ValueError("model is required")

    def generate(self, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.prompt_length", len(prompt))

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._extra_request_kwargs(),
            )

            if not response.choices:
                raise ValueError("LLM returned an empty response")

            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError("LLM returned an empty response")

            text = text.strip()
            span.set_attribute("file_agent.response_length", len(text))
            span.set_attribute("file_agent.response", text)

            usage = getattr(response, "usage", None)
            if usage is not None:
                span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

            logger.info(
                "LLM %s generated %d char(s) response (%d char(s) prompt)",
                self.model,
                len(text),
                len(prompt),
            )
            return text

    def _extra_request_kwargs(self) -> dict[str, Any]:
        if self.enable_thinking is None:
            return {}
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}}
