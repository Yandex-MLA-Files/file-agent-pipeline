import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from file_agent.llm.base import LLMClient
from file_agent.llm.openai_compatible import OpenAICompatibleClient


DEFAULT_LLM_BACKEND = "yandex"
DEFAULT_YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_YANDEX_MODEL = "qwen3.6-35b-a3b"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def create_llm_client(
    env_file: str | Path | None = ".env",
    load_env: bool = True,
    session: requests.Session | None = None,
) -> LLMClient:
    if load_env:
        load_dotenv(env_file)

    backend = os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND).strip().lower()

    if backend == "yandex":
        return _create_yandex_client(session=session)

    if backend == "local":
        return _create_local_client(session=session)

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def _create_yandex_client(
    session: requests.Session | None = None,
) -> OpenAICompatibleClient:
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

    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        auth_scheme="Api-Key",
        model=_build_yandex_model_uri(folder_id=folder_id, model=model),
        session=session,
    )


def _create_local_client(
    session: requests.Session | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=_getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL),
        api_key=_getenv("LOCAL_LLM_API_KEY"),
        auth_scheme=_getenv("LOCAL_LLM_AUTH_SCHEME", "Bearer"),
        model=_getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
        session=session,
        require_api_key=False,
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
