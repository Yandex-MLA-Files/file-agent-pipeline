import logging
import math
import os

from .base import VLMClient
from .openai_compatible import OpenAICompatibleVLMClient
from .smolvlm import DEFAULT_MODEL as DEFAULT_LOCAL_VLM
from .smolvlm import SmolVLMClient

logger = logging.getLogger(__name__)

DEFAULT_VLM_BACKEND = "off"
DEFAULT_VLM_MAX_TOKENS = 1000
DEFAULT_VLM_TEMPERATURE = 0.2
DEFAULT_VLM_TIMEOUT_SECONDS = 120.0
DEFAULT_VLM_MAX_RETRIES = 0


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
            max_tokens=_get_positive_int("VLM_MAX_TOKENS", DEFAULT_VLM_MAX_TOKENS),
            temperature=_get_float("VLM_TEMPERATURE", DEFAULT_VLM_TEMPERATURE),
            timeout_seconds=_get_positive_float(
                "VLM_TIMEOUT_SECONDS",
                DEFAULT_VLM_TIMEOUT_SECONDS,
            ),
            max_retries=_get_non_negative_int(
                "VLM_MAX_RETRIES",
                DEFAULT_VLM_MAX_RETRIES,
            ),
            enable_thinking=_get_optional_bool("VLM_ENABLE_THINKING"),
        )

    logger.warning("Unknown VLM_BACKEND=%r; VLM disabled.", backend)
    return None


def _get_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _get_non_negative_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _get_positive_float(name: str, default: float) -> float:
    value = _get_float(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _get_optional_bool(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
