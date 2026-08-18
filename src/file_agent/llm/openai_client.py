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
CONTEXT_SAFETY_MARGIN_TOKENS = 256
MIN_MAX_TOKENS = 256


class OpenAILLMClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
        retry_delay_seconds: float = DEFAULT_EMPTY_RESPONSE_RETRY_DELAY_SECONDS,
        context_length: int | None = None,
    ) -> None:
        self.client = client
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.empty_response_retries = empty_response_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.context_length = context_length

        if not self.model:
            raise ValueError("model is required")

    def _effective_max_tokens(self, messages: list[dict[str, Any]]) -> int:

        if self.context_length is None:
            return self.max_tokens
        estimated_prompt_tokens = len(json.dumps(messages, ensure_ascii=False)) // 2
        available = self.context_length - estimated_prompt_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
        return max(MIN_MAX_TOKENS, min(self.max_tokens, available))

    def generate(self, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.prompt_length", len(prompt))

            request_messages = [{"role": "user", "content": prompt}]
            attempts = self.empty_response_retries + 1
            for attempt in range(1, attempts + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=self.temperature,
                    max_tokens=self._effective_max_tokens(request_messages),
                )

                message = response.choices[0].message if response.choices else None
                text = message.content if message is not None else None
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

                reasoning = None
                if message is not None:
                    reasoning = getattr(message, "reasoning", None) or getattr(
                        message, "reasoning_content", None
                    )
                hint = (
                    f" - reasoning field had {len(reasoning)} char(s), likely truncated by "
                    "max_tokens before reaching a final answer"
                    if reasoning
                    else ""
                )
                logger.warning(
                    "LLM %s returned an empty response (attempt %d/%d)%s",
                    self.model,
                    attempt,
                    attempts,
                    hint,
                )
                if attempt < attempts:
                    time.sleep(self.retry_delay_seconds)

            raise ValueError(f"LLM returned an empty response after {attempts} attempt(s){hint}")

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
                    max_tokens=self._effective_max_tokens(messages),
                )

                message = response.choices[0].message if response.choices else None
                text = message.content if message is not None else None
                raw_tool_calls = getattr(message, "tool_calls", None) or []

                reasoning = getattr(message, "reasoning", None) or getattr(
                    message, "reasoning_content", None
                )

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
                    if reasoning:
                        span.set_attribute("file_agent.reasoning", reasoning)

                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                        span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

                    return ToolCallResponse(
                        content=text,
                        tool_calls=tool_calls,
                        reasoning=reasoning,
                        usage=_usage_dict(usage),
                    )

                hint = (
                    f" - reasoning field had {len(reasoning)} char(s), likely truncated by "
                    "max_tokens (possibly capped tighter than usual by a large prompt) before "
                    "reaching a final answer or tool call"
                    if reasoning
                    else ""
                )
                logger.warning(
                    "LLM %s returned an empty response with no tool calls (attempt %d/%d)%s",
                    self.model,
                    attempt,
                    attempts,
                    hint,
                )
                if attempt < attempts:
                    time.sleep(self.retry_delay_seconds)

            raise ValueError(f"LLM returned an empty response after {attempts} attempt(s){hint}")


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"_parse_error": f"invalid JSON arguments: {raw_arguments!r}"}
    if isinstance(parsed, dict):
        return parsed
    return {"_parse_error": f"expected a JSON object, got {parsed!r}"}
