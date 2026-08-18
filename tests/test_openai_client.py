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
        choices.append(SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None)))
    return SimpleNamespace(choices=choices)


def _tool_call_completion(content, tool_calls, **message_extra):
    message = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments_json),
            )
            for call_id, name, arguments_json in tool_calls
        ],
        **message_extra,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


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


def test_generate_raises_with_attempt_count_and_reasoning_hint():
    # The diagnostic (attempt count, reasoning length) must travel with the
    # exception itself, not just a log line - otherwise a UI that only shows
    # the raised exception (e.g. a trace viewer) gives no clue why.
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning="thinking a lot"))]
    )
    client = OpenAILLMClient(
        client=FakeOpenAI(response),
        model="test-model",
        empty_response_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(ValueError, match=r"after 1 attempt\(s\).*reasoning field had 14 char"):
        client.generate("Question")


def test_generate_with_tools_parses_tool_calls_from_the_response():
    openai_client = FakeOpenAI(
        _tool_call_completion(None, [("call-1", "search_documents", '{"query": "foo"}')])
    )
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(
        messages=[{"role": "user", "content": "Question"}],
        tools=[{"name": "search_documents", "description": "...", "parameters": {}}],
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "search_documents"
    assert response.tool_calls[0].arguments == {"query": "foo"}
    assert openai_client.completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Question"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "...",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    ]


def test_generate_with_tools_parses_final_answer_without_tool_calls():
    openai_client = FakeOpenAI(_chat_completion("Final answer"))
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[{"role": "user", "content": "Q"}], tools=[])

    assert response.content == "Final answer"
    assert response.tool_calls == []


def test_generate_with_tools_treats_content_none_with_tool_calls_as_non_empty():
    """A None-content, populated-tool_calls response is valid and must not retry."""
    openai_client = ScriptedOpenAI([_tool_call_completion(None, [("call-1", "calculate", "{}")])])
    client = OpenAILLMClient(
        client=openai_client, model="test-model", empty_response_retries=2, retry_delay_seconds=0
    )

    response = client.generate_with_tools(messages=[], tools=[])

    assert len(openai_client.completions.calls) == 1
    assert response.tool_calls[0].name == "calculate"


def test_generate_with_tools_retries_on_genuinely_empty_response():
    openai_client = ScriptedOpenAI(
        [
            _tool_call_completion(None, []),
            _chat_completion("Final answer"),
        ]
    )
    client = OpenAILLMClient(
        client=openai_client, model="test-model", empty_response_retries=2, retry_delay_seconds=0
    )

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.content == "Final answer"
    assert len(openai_client.completions.calls) == 2


def test_generate_with_tools_raises_after_exhausting_retries():
    openai_client = ScriptedOpenAI([_tool_call_completion(None, []) for _ in range(3)])
    client = OpenAILLMClient(
        client=openai_client, model="test-model", empty_response_retries=2, retry_delay_seconds=0
    )

    with pytest.raises(ValueError, match="empty response"):
        client.generate_with_tools(messages=[], tools=[])


def test_generate_with_tools_raises_with_attempt_count_and_reasoning_hint():
    openai_client = FakeOpenAI(_tool_call_completion(None, [], reasoning="thinking a lot"))
    client = OpenAILLMClient(
        client=openai_client, model="test-model", empty_response_retries=0, retry_delay_seconds=0
    )

    with pytest.raises(ValueError, match=r"after 1 attempt\(s\).*reasoning field had 14 char"):
        client.generate_with_tools(messages=[], tools=[])


def test_generate_with_tools_captures_reasoning_field():
    openai_client = FakeOpenAI(
        _tool_call_completion(
            None, [("call-1", "calculate", "{}")], reasoning="Let me think about this."
        )
    )
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.reasoning == "Let me think about this."


def test_generate_with_tools_falls_back_to_reasoning_content_field():
    """Some reasoning-parser backends use reasoning_content instead of reasoning."""
    openai_client = FakeOpenAI(
        _tool_call_completion(
            None, [("call-1", "calculate", "{}")], reasoning_content="Thinking..."
        )
    )
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.reasoning == "Thinking..."


def test_generate_with_tools_reasoning_defaults_to_none():
    openai_client = FakeOpenAI(_tool_call_completion(None, [("call-1", "calculate", "{}")]))
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.reasoning is None


def test_generate_with_tools_captures_token_usage():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id="call-1", function=SimpleNamespace(name="calculate", arguments="{}"))
        ],
    )
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    openai_client = FakeOpenAI(
        SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
    )
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_generate_with_tools_usage_defaults_to_none_without_a_usage_field():
    openai_client = FakeOpenAI(_tool_call_completion(None, [("call-1", "calculate", "{}")]))
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert response.usage is None


def test_generate_with_tools_turns_malformed_arguments_into_a_parse_error():
    openai_client = FakeOpenAI(
        _tool_call_completion(None, [("call-1", "calculate", "not valid json")])
    )
    client = OpenAILLMClient(client=openai_client, model="test-model")

    response = client.generate_with_tools(messages=[], tools=[])

    assert "_parse_error" in response.tool_calls[0].arguments


def test_max_tokens_is_unbounded_without_a_configured_context_length():
    openai_client = FakeOpenAI(_chat_completion("answer"))
    client = OpenAILLMClient(client=openai_client, model="test-model", max_tokens=5000)

    client.generate("Question")

    assert openai_client.completions.calls[0]["max_tokens"] == 5000


def test_generate_with_tools_caps_max_tokens_to_fit_a_large_prompt():
    """Regression test: a fixed max_tokens request fails outright (400) once
    prompt + max_tokens exceeds the model's real context window - a long chat
    history or several rounds of accumulated tool-call results can reach that
    on their own. max_tokens must shrink to fit instead."""
    openai_client = FakeOpenAI(_tool_call_completion(None, [("call-1", "calculate", "{}")]))
    client = OpenAILLMClient(
        client=openai_client, model="test-model", max_tokens=5000, context_length=1000
    )
    large_messages = [{"role": "user", "content": "x" * 3000}]  # ~1500 estimated tokens

    client.generate_with_tools(messages=large_messages, tools=[])

    sent_max_tokens = openai_client.completions.calls[0]["max_tokens"]
    assert sent_max_tokens < 5000
    assert sent_max_tokens >= 256  # MIN_MAX_TOKENS floor


def test_effective_max_tokens_never_drops_below_the_floor():
    openai_client = FakeOpenAI(_chat_completion("answer"))
    client = OpenAILLMClient(
        client=openai_client, model="test-model", max_tokens=5000, context_length=100
    )
    # A prompt that alone already exceeds context_length.
    huge_messages = [{"role": "user", "content": "x" * 10000}]

    sent_max_tokens = client._effective_max_tokens(huge_messages)

    assert sent_max_tokens == 256  # MIN_MAX_TOKENS floor, never zero or negative
