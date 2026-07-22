from abc import ABC, abstractmethod

from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class MockVLMClient(VLMClient):
    """Fallback client used when no VLM service is available (graceful degradation)."""

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        return (
            "[VLM unavailable]: the image was not analyzed. Check that the VLM server is reachable."
        )
