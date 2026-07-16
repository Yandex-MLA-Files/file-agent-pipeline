from abc import ABC, abstractmethod
from PIL import Image

class VLMClient(ABC):
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        pass

class MockVLMClient(VLMClient):
    """Fallback-клиент, если VLM-сервис недоступен (Graceful Degradation)"""
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        return "[VLM Unavailable]: Изображение не было проанализировано. Проверьте доступность VLM-сервера."