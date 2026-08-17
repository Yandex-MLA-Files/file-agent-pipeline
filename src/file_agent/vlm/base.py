from abc import ABC, abstractmethod

from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        """Answer ``prompt`` about ``image``; ``max_tokens`` overrides the client default."""
        raise NotImplementedError


class MockVLMClient(VLMClient):
    """Fallback client used when no VLM service is available (graceful degradation)."""

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        return (
            "[VLM unavailable]: the image was not analyzed. Check that the VLM server is reachable."
        )
