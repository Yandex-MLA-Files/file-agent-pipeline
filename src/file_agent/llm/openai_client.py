import logging
import time

from openai import OpenAI

from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_EMPTY_RESPONSE_RETRIES = 2
DEFAULT_EMPTY_RESPONSE_RETRY_DELAY_SECONDS = 1.0


class OpenAILLMClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
        retry_delay_seconds: float = DEFAULT_EMPTY_RESPONSE_RETRY_DELAY_SECONDS,
    ) -> None:
        self.client = client
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.empty_response_retries = empty_response_retries
        self.retry_delay_seconds = retry_delay_seconds

        if not self.model:
            raise ValueError("model is required")

    def generate(self, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.prompt_length", len(prompt))

            # Some backends occasionally return a 200 OK with empty content
            # (observed with deepseek-v4-flash on Yandex AI Studio) - retrying
            # the same request is usually enough, so don't fail the whole
            # answer over one flaky response.
            attempts = self.empty_response_retries + 1
            for attempt in range(1, attempts + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                text = response.choices[0].message.content if response.choices else None
                if text and text.strip():
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

                logger.warning(
                    "LLM %s returned an empty response (attempt %d/%d)",
                    self.model,
                    attempt,
                    attempts,
                )
                if attempt < attempts:
                    time.sleep(self.retry_delay_seconds)

            raise ValueError("LLM returned an empty response")
