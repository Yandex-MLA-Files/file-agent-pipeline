import logging
import os

from .base import VLMClient
from .openai_compatible import OpenAICompatibleVLMClient
from .smolvlm import DEFAULT_MODEL as DEFAULT_LOCAL_VLM
from .smolvlm import SmolVLMClient

logger = logging.getLogger(__name__)

# The chat model served for answering (Qwen3.5 on vLLM) is multimodal, so by
# default the *same* endpoint describes figures and transcribes scanned pages —
# no second model to deploy, no extra GPU memory.
DEFAULT_VLM_BACKEND = "llm"
DEFAULT_VLM_TIMEOUT_SECONDS = 180.0
DEFAULT_VLM_MAX_TOKENS = 1200


def create_vlm_client() -> VLMClient | None:
    """Build the VLM client selected by the ``VLM_BACKEND`` environment variable.

    - ``llm`` (default): reuse the answering model's OpenAI-compatible endpoint
      (``LOCAL_LLM_BASE_URL`` / ``LOCAL_LLM_MODEL`` / ``LOCAL_LLM_API_KEY``) as
      the vision model. Works when that model is multimodal (Qwen3.5, Qwen-VL,
      ...); requires no additional deployment. Thinking is disabled for vision
      calls unless ``VLM_ENABLE_THINKING`` says otherwise — transcription does
      not benefit from it and it is several times slower.
    - ``openai``: any *separate* OpenAI-compatible vision endpoint (Ollama
      ``qwen2.5-vl``, vLLM, a cloud API) configured via ``VLM_BASE_URL`` /
      ``VLM_MODEL`` / ``VLM_API_KEY``.
    - ``smolvlm``: local SmolVLM-256M via transformers. No server, no API key,
      CPU-friendly; model downloads once (~500 MB). Lowest quality.
      Override the checkpoint with ``VLM_LOCAL_MODEL``.
    - ``off``: figure description disabled — parsing stays fully offline.

    Returns None when disabled or misconfigured — callers skip enhancement.
    """
    backend = os.getenv("VLM_BACKEND", DEFAULT_VLM_BACKEND).strip().lower()

    if backend in ("", "off", "none", "disabled"):
        return None

    if backend == "smolvlm":
        return SmolVLMClient(model_name=os.getenv("VLM_LOCAL_MODEL", DEFAULT_LOCAL_VLM))

    if backend == "llm":
        base_url = os.getenv("VLM_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL")
        model = os.getenv("VLM_MODEL") or os.getenv("LOCAL_LLM_MODEL")
        if not base_url or not model:
            logger.info(
                "VLM_BACKEND=llm needs LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL (or VLM_BASE_URL "
                "and VLM_MODEL); figure description disabled."
            )
            return None
        api_key = os.getenv("VLM_API_KEY") or os.getenv("LOCAL_LLM_API_KEY") or "not-used"
        return _build_openai_client(base_url, model, api_key)

    if backend == "openai":
        base_url = os.getenv("VLM_BASE_URL")
        model = os.getenv("VLM_MODEL")
        if not base_url or not model:
            logger.warning("VLM_BACKEND=openai requires VLM_BASE_URL and VLM_MODEL; VLM disabled.")
            return None
        return _build_openai_client(base_url, model, os.getenv("VLM_API_KEY", "dummy"))

    logger.warning("Unknown VLM_BACKEND=%r; VLM disabled.", backend)
    return None


def _build_openai_client(base_url: str, model: str, api_key: str) -> OpenAICompatibleVLMClient:
    thinking_raw = (os.getenv("VLM_ENABLE_THINKING") or "false").strip().lower()
    enable_thinking = thinking_raw in {"1", "true", "yes", "on"}
    return OpenAICompatibleVLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=int(os.getenv("VLM_MAX_TOKENS", DEFAULT_VLM_MAX_TOKENS)),
        timeout=float(os.getenv("VLM_TIMEOUT_SECONDS", DEFAULT_VLM_TIMEOUT_SECONDS)),
        enable_thinking=enable_thinking,
    )
