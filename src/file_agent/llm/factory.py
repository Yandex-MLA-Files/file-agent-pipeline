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
) -> LLMClient:
    if load_env:
        load_dotenv(env_file)

    backend = os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND).strip().lower()

    if backend == "yandex":
        return _create_yandex_client()

    if backend == "local":
        return _create_local_client()

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def _create_yandex_client() -> OpenAILLMClient:
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
        **_generation_settings(),
    )


def _create_local_client() -> OpenAILLMClient:
    client = _create_openai_client(
        api_key=_getenv("LOCAL_LLM_API_KEY") or LOCAL_API_KEY_PLACEHOLDER,
        base_url=_getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
    )
    return OpenAILLMClient(
        client=client,
        model=_getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
        **_generation_settings(),
    )


def _generation_settings() -> dict[str, object]:
    """Sampling settings shared by every backend, overridable from the environment.

    ``LLM_ENABLE_THINKING`` matters for reasoning models served with a reasoning
    parser (Qwen3.5 on vLLM): left unset the server default applies, ``false``
    turns thinking off (fast, deterministic answers for batch evaluation),
    ``true`` forces it on.
    """
    settings: dict[str, object] = {
        "temperature": _float_env("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        "max_tokens": _int_env("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    }
    thinking = _bool_env("LLM_ENABLE_THINKING")
    if thinking is not None:
        settings["enable_thinking"] = thinking
    return settings


def _create_openai_client(
    api_key: str,
    base_url: str | None,
    project: str | None = None,
) -> OpenAI:
    if not base_url:
        raise ValueError("base_url is required")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        project=project,
        timeout=_float_env("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_retries=_int_env("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
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


def _float_env(name: str, default: float) -> float:
    raw = _getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw = _getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_env(name: str) -> bool | None:
    raw = _getenv(name)
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")
