from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from file_agent.agent_tools import search_documents
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


def _chat_completion(
    content: str | None,
    include_choice: bool = True,
    tool_calls=None,
):
    choices = []
    if include_choice:
        choices.append(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls or [],
                )
            )
        )
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


def test_generate_passes_qwen_thinking_setting():
    openai_client = FakeOpenAI(_chat_completion("Generated answer"))
    client = OpenAILLMClient(
        client=openai_client,
        model="test-model",
        enable_thinking=False,
    )

    client.generate("Question")

    assert openai_client.completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


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
    )

    with pytest.raises(ValueError, match="empty response"):
        client.generate("Question")


def test_chat_with_tools_converts_messages_tools_and_tool_calls():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="search_documents",
            arguments='{"query":"project deadline","top_k":3}',
        ),
    )
    openai_client = FakeOpenAI(_chat_completion(None, tool_calls=[tool_call]))
    client = OpenAILLMClient(
        client=openai_client,
        model="test-model",
        temperature=0.1,
        max_tokens=128,
        enable_thinking=False,
    )

    message = client.chat_with_tools(
        messages=[
            SystemMessage(content="Use document tools."),
            HumanMessage(content="When is the deadline?"),
        ],
        tools=[search_documents],
    )

    assert message.content == ""
    assert message.tool_calls == [
        {
            "name": "search_documents",
            "args": {"query": "project deadline", "top_k": 3},
            "id": "call-1",
            "type": "tool_call",
        }
    ]
    request = openai_client.completions.calls[0]
    assert request["messages"] == [
        {"role": "system", "content": "Use document tools."},
        {"role": "user", "content": "When is the deadline?"},
    ]
    assert request["tools"][0]["function"]["name"] == "search_documents"
    assert set(request["tools"][0]["function"]["parameters"]["properties"]) == {
        "query",
        "top_k",
        "source_file",
    }
    assert request["tool_choice"] == "auto"
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_chat_with_tools_can_force_a_final_response_without_tools():
    openai_client = FakeOpenAI(_chat_completion(" Final answer "))
    client = OpenAILLMClient(client=openai_client, model="test-model")

    message = client.chat_with_tools(
        messages=[HumanMessage(content="Question")],
        tools=[],
    )

    assert message.content == "Final answer"
    request = openai_client.completions.calls[0]
    assert "tools" not in request
    assert "tool_choice" not in request


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_chat_with_tools_rejects_invalid_tool_arguments(arguments):
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="search_documents", arguments=arguments),
    )
    client = OpenAILLMClient(
        client=FakeOpenAI(_chat_completion(None, tool_calls=[tool_call])),
        model="test-model",
    )

    with pytest.raises(ValueError, match="invalid arguments"):
        client.chat_with_tools(
            messages=[HumanMessage(content="Question")],
            tools=[search_documents],
        )
