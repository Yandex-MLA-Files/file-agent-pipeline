from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool


class EmptyLLMResponseError(ValueError):
    """Raised when an LLM request succeeds but returns no usable assistant message."""


ToolChoice = Literal["auto", "required"]


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@runtime_checkable
class ToolCallingLLMClient(LLMClient, Protocol):
    def chat_with_tools(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
        tool_choice: ToolChoice = "auto",
    ) -> AIMessage:
        """Return an assistant message that may contain native tool calls."""
        raise NotImplementedError
