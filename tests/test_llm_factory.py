import pytest

from file_agent.llm.factory import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_YANDEX_BASE_URL,
    DEFAULT_YANDEX_MODEL,
    LOCAL_API_KEY_PLACEHOLDER,
    create_llm_client,
)
from file_agent.llm.openai_client import OpenAILLMClient


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    monkeypatch.setattr("file_agent.llm.factory.OpenAI", FakeOpenAI)


def test_factory_creates_yandex_client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.delenv("YANDEX_MODEL", raising=False)
    monkeypatch.delenv("YANDEX_BASE_URL", raising=False)

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAILLMClient)
    assert client.model == f"gpt://folder-id/{DEFAULT_YANDEX_MODEL}"
    assert client.client.kwargs == {
        "api_key": "api-key",
        "base_url": DEFAULT_YANDEX_BASE_URL,
        "project": "folder-id",
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_MAX_RETRIES,
    }


def test_factory_keeps_full_yandex_model_uri(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "yandex")
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.setenv("YANDEX_MODEL", "gpt://folder/custom-qwen")

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAILLMClient)
    assert client.model == "gpt://folder/custom-qwen"
    assert client.client.kwargs["project"] is None


def test_factory_creates_local_client_without_configured_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)

    client = create_llm_client(load_env=False)

    assert isinstance(client, OpenAILLMClient)
    assert client.model == DEFAULT_LOCAL_MODEL
    assert client.client.kwargs == {
        "api_key": LOCAL_API_KEY_PLACEHOLDER,
        "base_url": DEFAULT_LOCAL_BASE_URL,
        "project": None,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": DEFAULT_MAX_RETRIES,
    }


def test_factory_uses_configured_local_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-api-key")

    client = create_llm_client(load_env=False)

    assert client.client.kwargs["api_key"] == "local-api-key"


def test_factory_rejects_empty_local_base_url(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "  ")

    with pytest.raises(ValueError, match="base_url"):
        create_llm_client(load_env=False)


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
