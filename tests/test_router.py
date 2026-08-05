import pytest

from file_agent.router import QueryType, build_router_prompt, classify_query


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.mark.parametrize(
    "response, expected",
    [
        ('{"query_type": "simple"}', QueryType.SIMPLE),
        ('{"query_type": "complex"}', QueryType.COMPLEX),
        ('{"query_type": "tool"}', QueryType.TOOL),
        ('{"query_type": "TOOL"}', QueryType.TOOL),
        ('Ответ: {"query_type": "complex"} спасибо', QueryType.COMPLEX),
    ],
)
def test_classify_query_parses_valid_responses(response, expected):
    llm_client = StubLLM(response)

    result = classify_query("Сколько всего строк в таблице?", llm_client)

    assert result == expected
    assert len(llm_client.prompts) == 1


@pytest.mark.parametrize(
    "response",
    [
        "не json вообще",
        '{"query_type": "unknown-category"}',
        "{}",
        "",
    ],
)
def test_classify_query_defaults_to_simple_on_bad_response(response):
    llm_client = StubLLM(response)

    result = classify_query("Что такое проект?", llm_client)

    assert result == QueryType.SIMPLE


def test_classify_query_defaults_to_simple_when_llm_call_raises():
    class FailingLLM:
        def generate(self, prompt: str) -> str:
            raise ValueError("LLM returned an empty response")

    result = classify_query("Что такое проект?", FailingLLM())

    assert result == QueryType.SIMPLE


def test_build_router_prompt_includes_question_and_categories():
    prompt = build_router_prompt("Сравни выручку за 2022 и 2023 год")

    assert "Сравни выручку за 2022 и 2023 год" in prompt
    assert "simple" in prompt
    assert "complex" in prompt
    assert "tool" in prompt
