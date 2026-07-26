import logging
import os

from .base import VLMClient
from .openai_compatible import OpenAICompatibleVLMClient
from .smolvlm import DEFAULT_MODEL as DEFAULT_LOCAL_VLM
from .smolvlm import SmolVLMClient

logger = logging.getLogger(__name__)

DEFAULT_VLM_BACKEND = "off"


def create_vlm_client() -> VLMClient | None:
    """Build the VLM client selected by the ``VLM_BACKEND`` environment variable.

    - ``off`` (default): figure description disabled — parsing stays fully
      offline and free.
    - ``smolvlm``: local SmolVLM-256M via transformers. Cheapest working option:
      no server, no API key, CPU-friendly; model downloads once (~500 MB).
      Override the checkpoint with ``VLM_LOCAL_MODEL``.
    - ``openai``: any OpenAI-compatible vision endpoint (Ollama ``qwen2.5-vl``,
      vLLM, a cloud API) configured via ``VLM_BASE_URL`` / ``VLM_MODEL`` /
      ``VLM_API_KEY``. Best quality; cost depends on the endpoint.

    Returns None when disabled or misconfigured — callers skip enhancement.
    """
    backend = os.getenv("VLM_BACKEND", DEFAULT_VLM_BACKEND).strip().lower()

    if backend in ("", "off", "none", "disabled"):
        return None

    if backend == "smolvlm":
        return SmolVLMClient(model_name=os.getenv("VLM_LOCAL_MODEL", DEFAULT_LOCAL_VLM))

    if backend == "openai":
        base_url = os.getenv("VLM_BASE_URL")
        model = os.getenv("VLM_MODEL")
        if not base_url or not model:
            logger.warning("VLM_BACKEND=openai requires VLM_BASE_URL and VLM_MODEL; VLM disabled.")
            return None
        return OpenAICompatibleVLMClient(
            base_url=base_url,
            model=model,
            api_key=os.getenv("VLM_API_KEY", "dummy"),
        )

    logger.warning("Unknown VLM_BACKEND=%r; VLM disabled.", backend)
    return None
