import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from file_agent.llm.base import LLMClient
from file_agent.llm.openai_client import OpenAILLMClient

DEFAULT_LLM_BACKEND = "yandex"
DEFAULT_YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_YANDEX_MODEL = "qwen3.6-35b-a3b"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
LOCAL_API_KEY_PLACEHOLDER = "not-used"


def create_llm_client(
    env_file: str | Path | None = ".env",
    load_env: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMClient:
    if load_env:
        load_dotenv(env_file)

    backend = os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND).strip().lower()

    if backend == "yandex":
        return _create_yandex_client(temperature=temperature, max_tokens=max_tokens)

    if backend == "local":
        return _create_local_client(temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def _create_yandex_client(temperature: float, max_tokens: int) -> OpenAILLMClient:
    api_key = _getenv("YANDEX_API_KEY")
    folder_id = _getenv("YANDEX_FOLDER_ID")
    model = _getenv("YANDEX_MODEL", DEFAULT_YANDEX_MODEL)
    base_url = _getenv("YANDEX_BASE_URL", DEFAULT_YANDEX_BASE_URL)

    if not api_key:
        raise ValueError("YANDEX_API_KEY is required for LLM_BACKEND=yandex")
    if not model:
        raise ValueError("YANDEX_MODEL is required for LLM_BACKEND=yandex")
    if not folder_id and not model.startswith("gpt://"):
        raise ValueError("YANDEX_FOLDER_ID is required for LLM_BACKEND=yandex")

    client = _create_openai_client(
        api_key=api_key,
        base_url=base_url,
        project=folder_id,
    )
    return OpenAILLMClient(
        client=client,
        model=_build_yandex_model_uri(folder_id=folder_id, model=model),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_local_client(temperature: float, max_tokens: int) -> OpenAILLMClient:
    client = _create_openai_client(
        api_key=_getenv("LOCAL_LLM_API_KEY") or LOCAL_API_KEY_PLACEHOLDER,
        base_url=_getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
        timeout_seconds=_get_positive_float(
            "LOCAL_LLM_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        ),
        max_retries=_get_non_negative_int(
            "LOCAL_LLM_MAX_RETRIES",
            DEFAULT_MAX_RETRIES,
        ),
    )
    return OpenAILLMClient(
        client=client,
        model=_getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=_get_optional_bool("LOCAL_LLM_ENABLE_THINKING"),
    )


def _create_openai_client(
    api_key: str,
    base_url: str | None,
    project: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    if not base_url:
        raise ValueError("base_url is required")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        project=project,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _build_yandex_model_uri(folder_id: str | None, model: str) -> str:
    if model.startswith("gpt://"):
        return model

    return f"gpt://{folder_id}/{model}"


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None

    return value.strip()


def _get_optional_bool(name: str) -> bool | None:
    value = _getenv(name)
    if not value:
        return None

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _get_positive_float(name: str, default: float) -> float:
    raw = _getenv(name)
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _get_non_negative_int(name: str, default: int) -> int:
    raw = _getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
