import logging
import os

from .base import VLMClient
from .openai_compatible import OpenAICompatibleVLMClient
from .smolvlm import DEFAULT_MODEL as DEFAULT_LOCAL_VLM
from .smolvlm import SmolVLMClient

logger = logging.getLogger(__name__)

DEFAULT_VLM_BACKEND = "off"
DEFAULT_VLM_MAX_TOKENS = 1500


def create_vlm_client() -> VLMClient | None:

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
            max_tokens=int(os.getenv("VLM_MAX_TOKENS", str(DEFAULT_VLM_MAX_TOKENS))),
        )

    logger.warning("Unknown VLM_BACKEND=%r; VLM disabled.", backend)
    return None
