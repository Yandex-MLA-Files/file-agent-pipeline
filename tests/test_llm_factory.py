import pytest

from file_agent.llm.factory import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_YANDEX_BASE_URL,
    DEFAULT_YANDEX_MODEL,
    create_llm_client,
)
from file_agent.llm.openai_compatible import OpenAICompatibleClient


def test_factory_creates_yandex_client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.delenv("YANDEX_MODEL", raising=False)
    monkeypatch.delenv("YANDEX_BASE_URL", raising=False)

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == DEFAULT_YANDEX_BASE_URL
    assert client.api_key == "api-key"
    assert client.auth_scheme == "Api-Key"
    assert client.model == f"gpt://folder-id/{DEFAULT_YANDEX_MODEL}"


def test_factory_keeps_full_yandex_model_uri(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.setenv("YANDEX_MODEL", "gpt://folder/custom-qwen")

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "gpt://folder/custom-qwen"


def test_factory_creates_local_client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == DEFAULT_LOCAL_BASE_URL
    assert client.model == DEFAULT_LOCAL_MODEL
    assert client.require_api_key is False


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_BACKEND"):
        create_llm_client(load_env=False)


def test_factory_requires_yandex_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "yandex")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")

    with pytest.raises(ValueError, match="YANDEX_API_KEY"):
        create_llm_client(load_env=False)
