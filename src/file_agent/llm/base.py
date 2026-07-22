from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
