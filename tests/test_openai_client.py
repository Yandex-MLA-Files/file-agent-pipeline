from types import SimpleNamespace

import pytest

from file_agent.llm.openai_client import OpenAILLMClient


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    def __init__(self, response):
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


class ScriptedCompletions:
    """Returns each response in order, one per create() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ScriptedOpenAI:
    def __init__(self, responses):
        self.completions = ScriptedCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _chat_completion(content: str | None, include_choice: bool = True):
    choices = []
    if include_choice:
        choices.append(SimpleNamespace(message=SimpleNamespace(content=content)))
    return SimpleNamespace(choices=choices)


def test_generate_uses_openai_chat_completions():
    openai_client = FakeOpenAI(_chat_completion("  Generated answer  "))
    client = OpenAILLMClient(
        client=openai_client,
        model="test-model",
        temperature=0.1,
        max_tokens=128,
    )

    answer = client.generate("Question")

    assert answer == "Generated answer"
    assert openai_client.completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Question"}],
            "temperature": 0.1,
            "max_tokens": 128,
        }
    ]


def test_client_requires_model():
    with pytest.raises(ValueError, match="model"):
        OpenAILLMClient(
            client=FakeOpenAI(_chat_completion("Generated answer")),
            model="  ",
        )


@pytest.mark.parametrize(
    ("response",),
    [
        (_chat_completion(None),),
        (_chat_completion("  "),),
        (_chat_completion("unused", include_choice=False),),
    ],
)
def test_generate_raises_clear_error_for_empty_response(response):
    client = OpenAILLMClient(
        client=FakeOpenAI(response),
        model="test-model",
        empty_response_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(ValueError, match="empty response"):
        client.generate("Question")


def test_generate_retries_after_empty_response_then_succeeds():
    openai_client = ScriptedOpenAI(
        [
            _chat_completion(""),
            _chat_completion("Generated answer"),
        ]
    )
    client = OpenAILLMClient(
        client=openai_client,
        model="test-model",
        empty_response_retries=2,
        retry_delay_seconds=0,
    )

    answer = client.generate("Question")

    assert answer == "Generated answer"
    assert len(openai_client.completions.calls) == 2


def test_generate_raises_after_exhausting_retries():
    openai_client = ScriptedOpenAI([_chat_completion("") for _ in range(3)])
    client = OpenAILLMClient(
        client=openai_client,
        model="test-model",
        empty_response_retries=2,
        retry_delay_seconds=0,
    )

    with pytest.raises(ValueError, match="empty response"):
        client.generate("Question")

    assert len(openai_client.completions.calls) == 3
