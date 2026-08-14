import json
import logging
import math
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI

from file_agent.llm.base import EmptyLLMResponseError, ToolChoice
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
        self.enable_thinking = enable_thinking
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        if not self.model:
            raise ValueError("model is required")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

    def generate(self, prompt: str) -> str:
        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.prompt_length", len(prompt))
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("langfuse.observation.model.name", self.model)
            span.set_attribute(
                "langfuse.observation.model.parameters",
                json.dumps(
                    {
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "enable_thinking": self.enable_thinking,
                    }
                ),
            )
            span.set_attribute(
                "langfuse.observation.input",
                json.dumps({"messages": [{"role": "user", "content": prompt}]}),
            )

            request = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            self._add_thinking_setting(request)
            response = self.client.chat.completions.create(**request)
            usage = getattr(response, "usage", None)
            self._record_usage(usage)

            if not response.choices:
                raise EmptyLLMResponseError("LLM returned an empty response")

            text = response.choices[0].message.content
            if not text or not text.strip():
                raise EmptyLLMResponseError("LLM returned an empty response")

            text = text.strip()
            span.set_attribute("file_agent.response_length", len(text))
            span.set_attribute("file_agent.response", text)
            span.set_attribute(
                "langfuse.observation.output",
                json.dumps({"role": "assistant", "content": text}),
            )

            if usage is not None:
                span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)
                span.set_attribute(
                    "langfuse.observation.usage_details",
                    json.dumps(
                        {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    ),
                )

            logger.info(
                "LLM %s generated %d char(s) response (%d char(s) prompt)",
                self.model,
                len(text),
                len(prompt),
            )
            return text

    def chat_with_tools(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
        tool_choice: ToolChoice = "auto",
    ) -> AIMessage:
        if tool_choice not in {"auto", "required"}:
            raise ValueError("tool_choice must be 'auto' or 'required'")
        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' needs at least one tool")

        request = {
            "model": self.model,
            "messages": convert_to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        self._add_thinking_setting(request)
        if tools:
            request["tools"] = [convert_to_openai_tool(tool) for tool in tools]
            request["tool_choice"] = tool_choice

        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.message_count", len(messages))
            span.set_attribute("file_agent.tool_count", len(tools))
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("langfuse.observation.model.name", self.model)
            span.set_attribute(
                "langfuse.observation.model.parameters",
                json.dumps(
                    {
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "enable_thinking": self.enable_thinking,
                    }
                ),
            )
            span.set_attribute(
                "langfuse.observation.input",
                json.dumps(request["messages"], ensure_ascii=False, default=str),
            )

            response = self.client.chat.completions.create(**request)
            usage = getattr(response, "usage", None)
            self._record_usage(usage)
            if not response.choices:
                raise EmptyLLMResponseError("LLM returned an empty response")

            message = response.choices[0].message
            content = (message.content or "").strip()
            tool_calls = []
            for tool_call in message.tool_calls or []:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"LLM returned invalid arguments for tool {tool_call.function.name}"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"LLM returned invalid arguments for tool {tool_call.function.name}"
                    )
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "args": arguments,
                        "id": tool_call.id,
                        "type": "tool_call",
                    }
                )

            if not content and not tool_calls:
                raise EmptyLLMResponseError("LLM returned an empty response")

            span.set_attribute("file_agent.response_length", len(content))
            span.set_attribute("file_agent.response", content)
            span.set_attribute("file_agent.tool_call_count", len(tool_calls))
            span.set_attribute(
                "langfuse.observation.output",
                json.dumps(
                    {"content": content, "tool_calls": tool_calls},
                    ensure_ascii=False,
                    default=str,
                ),
            )

            if usage is not None:
                span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)
                span.set_attribute(
                    "langfuse.observation.usage_details",
                    json.dumps(
                        {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    ),
                )

            logger.info(
                "LLM %s returned %d char(s) and %d tool call(s)",
                self.model,
                len(content),
                len(tool_calls),
            )
            return AIMessage(content=content, tool_calls=tool_calls)

    def _add_thinking_setting(self, request: dict) -> None:
        if self.enable_thinking is None:
            return
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    def _record_usage(self, usage: object | None) -> None:
        self.request_count += 1
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
