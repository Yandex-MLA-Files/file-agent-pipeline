import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI


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

        return text.strip()

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

        return AIMessage(content=content, tool_calls=tool_calls)
