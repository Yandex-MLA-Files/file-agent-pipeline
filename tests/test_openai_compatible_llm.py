import pytest

from file_agent.llm.openai_compatible import OpenAICompatibleClient


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


def test_generate_calls_openai_compatible_chat_completions():
    session = FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer",
                    }
                }
            ]
        }
    )
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1/",
        api_key="token",
        model="test-model",
        session=session,
        temperature=0.1,
        max_tokens=128,
    )

    answer = client.generate("Question")

    assert answer == "Generated answer"
    assert len(session.calls) == 1
    args, kwargs = session.calls[0]
    assert args[0] == "http://localhost:8000/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer token"
    assert kwargs["json"]["model"] == "test-model"
    assert kwargs["json"]["messages"] == [
        {"role": "user", "content": "Question"}
    ]
    assert kwargs["json"]["temperature"] == 0.1
    assert kwargs["json"]["max_tokens"] == 128


def test_client_supports_yandex_api_key_auth_scheme():
    session = FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer",
                    }
                }
            ]
        }
    )
    client = OpenAICompatibleClient(
        base_url="https://ai.api.cloud.yandex.net/v1",
        api_key="api-key",
        auth_scheme="Api-Key",
        model="gpt://folder/qwen",
        session=session,
    )

    client.generate("Question")

    _, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Api-Key api-key"


def test_client_can_skip_auth_for_local_endpoint():
    session = FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "content": "Generated answer",
                    }
                }
            ]
        }
    )
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="local-model",
        session=session,
        require_api_key=False,
    )

    client.generate("Question")

    _, kwargs = session.calls[0]
    assert "Authorization" not in kwargs["headers"]


def test_client_requires_api_key_by_default():
    with pytest.raises(ValueError, match="api_key"):
        OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
        )


def test_generate_raises_clear_error_for_empty_response():
    session = FakeSession({"choices": [{"message": {"content": "  "}}]})
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        api_key="token",
        model="test-model",
        session=session,
    )

    with pytest.raises(ValueError, match="empty response"):
        client.generate("Question")
