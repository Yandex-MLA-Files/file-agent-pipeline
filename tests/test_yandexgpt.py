import pytest

from file_agent.llm.yandexgpt import DEFAULT_YANDEX_MODEL, YandexGPTClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeResponse(self.payload)


def test_client_fails_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")

    with pytest.raises(ValueError, match="YANDEX_API_KEY"):
        YandexGPTClient(load_env=False)


def test_client_fails_when_folder_id_missing(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    with pytest.raises(ValueError, match="YANDEX_FOLDER_ID"):
        YandexGPTClient(load_env=False)


def test_client_uses_default_model(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.delenv("YANDEX_MODEL", raising=False)

    client = YandexGPTClient(load_env=False)

    assert client.model == DEFAULT_YANDEX_MODEL
    assert client.model_uri == f"gpt://folder-id/{DEFAULT_YANDEX_MODEL}/latest"


def test_generate_calls_mocked_api_client_and_returns_model_text(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.setenv("YANDEX_MODEL", "custom-model")
    session = FakeSession(
        {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "text": "Generated answer",
                        }
                    }
                ]
            }
        }
    )
    client = YandexGPTClient(session=session, load_env=False)

    answer = client.generate("Question")

    assert answer == "Generated answer"
    assert len(session.calls) == 1
    args, kwargs = session.calls[0]
    assert args[0].endswith("/foundationModels/v1/completion")
    assert kwargs["headers"]["Authorization"] == "Api-Key api-key"
    assert kwargs["json"]["modelUri"] == "gpt://folder-id/custom-model/latest"
    assert kwargs["json"]["messages"][0]["text"] == "Question"


def test_generate_raises_clear_error_for_empty_model_response(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "api-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    session = FakeSession(
        {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "text": "  ",
                        }
                    }
                ]
            }
        }
    )
    client = YandexGPTClient(session=session, load_env=False)

    with pytest.raises(ValueError, match="empty response"):
        client.generate("Question")
