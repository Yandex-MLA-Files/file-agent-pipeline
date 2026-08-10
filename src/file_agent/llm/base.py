from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallResponse:
    content: str | None
    tool_calls: list[ToolCall]


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ToolCallResponse:
        raise NotImplementedError
