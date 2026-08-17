from abc import ABC, abstractmethod

from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        """Answer ``prompt`` about ``image``; ``max_tokens`` overrides the client default."""
        raise NotImplementedError

    def describe_image_verbose(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int | None = None,
        max_image_side: int | None = None,
    ) -> tuple[str, str | None]:
        """Like :meth:`describe_image` but also report why generation stopped.

        Page transcription needs to know the difference between "the page ends
        here" and "the token budget ran out" — the latter is retried with a
        larger budget instead of being indexed as a truncated page. Backends
        that cannot report it return ``None``, and callers fall back to
        heuristics. ``max_image_side`` raises the client's downscale limit for
        text-dense pages.
        """
        return self.describe_image(image, prompt, max_tokens=max_tokens), None


class MockVLMClient(VLMClient):
    """Fallback client used when no VLM service is available (graceful degradation)."""

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        return (
            "[VLM unavailable]: the image was not analyzed. Check that the VLM server is reachable."
        )
