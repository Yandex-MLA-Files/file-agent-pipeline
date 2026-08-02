import json
import logging
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
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
    ) -> None:
        self.client = client
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

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

    def chat_with_tools(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
    ) -> AIMessage:
        request = {
            "model": self.model,
            "messages": convert_to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            request["tools"] = [convert_to_openai_tool(tool) for tool in tools]
            request["tool_choice"] = "auto"

        with tracer.start_as_current_span("file_agent.llm_generate") as span:
            span.set_attribute("file_agent.model", self.model)
            span.set_attribute("file_agent.message_count", len(messages))
            span.set_attribute("file_agent.tool_count", len(tools))

            response = self.client.chat.completions.create(**request)
            if not response.choices:
                raise ValueError("LLM returned an empty response")

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
                raise ValueError("LLM returned an empty response")

            span.set_attribute("file_agent.response_length", len(content))
            span.set_attribute("file_agent.response", content)
            span.set_attribute("file_agent.tool_call_count", len(tool_calls))

            usage = getattr(response, "usage", None)
            if usage is not None:
                span.set_attribute("file_agent.prompt_tokens", usage.prompt_tokens)
                span.set_attribute("file_agent.completion_tokens", usage.completion_tokens)

            logger.info(
                "LLM %s returned %d char(s) and %d tool call(s)",
                self.model,
                len(content),
                len(tool_calls),
            )
            return AIMessage(content=content, tool_calls=tool_calls)
