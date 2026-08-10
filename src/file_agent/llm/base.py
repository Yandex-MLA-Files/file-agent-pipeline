from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class ChatLLMClient(Protocol):
    """LLM client that accepts a multi-turn conversation.

    Messages follow the OpenAI chat format: a list of dicts with
    ``role`` (``system`` / ``user`` / ``assistant``) and ``content`` keys.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError
