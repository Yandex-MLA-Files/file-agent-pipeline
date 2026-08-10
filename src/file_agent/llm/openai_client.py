import json
import logging
import time
from typing import Any

from openai import OpenAI

from file_agent.llm.base import ToolCall, ToolCallResponse
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

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ToolCallResponse:
        with tracer.start_as_current_span("file_agent.llm_generate_with_tools") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.tool_count", len(tools))

            attempts = self.empty_response_retries + 1
            for attempt in range(1, attempts + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": tool} for tool in tools],
                    tool_choice=tool_choice,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                message = response.choices[0].message if response.choices else None
                text = message.content if message is not None else None
                raw_tool_calls = getattr(message, "tool_calls", None) or []

                # A valid tool-call response has content=None but populated
                # tool_calls - that is not the same as a genuinely empty
                # (failed) response, so it must not trigger a retry.
                is_empty = not raw_tool_calls and not (text and text.strip())
                if not is_empty:
                    tool_calls = [
                        ToolCall(
                            id=call.id,
                            name=call.function.name,
                            arguments=_parse_tool_arguments(call.function.arguments),
                        )
                        for call in raw_tool_calls
                    ]
                    span.set_attribute("file_agent.tool_calls_count", len(tool_calls))
                    if text:
                        span.set_attribute("file_agent.response_length", len(text))

                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                        span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

                    return ToolCallResponse(content=text, tool_calls=tool_calls)

                logger.warning(
                    "LLM %s returned an empty response with no tool calls (attempt %d/%d)",
                    self.model,
                    attempt,
                    attempts,
                )
                if attempt < attempts:
                    time.sleep(self.retry_delay_seconds)

            raise ValueError("LLM returned an empty response")


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"_parse_error": f"invalid JSON arguments: {raw_arguments!r}"}
    if isinstance(parsed, dict):
        return parsed
    return {"_parse_error": f"expected a JSON object, got {parsed!r}"}
