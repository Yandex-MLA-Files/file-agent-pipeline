import pytest

from file_agent.planner import MAX_SUBQUERIES, build_planner_prompt, plan_subqueries


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_plan_subqueries_parses_valid_response():
    llm_client = StubLLM('{"subqueries": ["Доходы за 2026 год", "Доходы за 2025 год"]}')

    result = plan_subqueries("Сравни доходы за 2026 и 2025 год", llm_client)

    assert result == ["Доходы за 2026 год", "Доходы за 2025 год"]
    assert len(llm_client.prompts) == 1


def test_plan_subqueries_strips_surrounding_prose():
    llm_client = StubLLM('Вот план: {"subqueries": ["A", "B"]} готово')

    assert plan_subqueries("A и B?", llm_client) == ["A", "B"]


def test_plan_subqueries_drops_blank_entries():
    llm_client = StubLLM('{"subqueries": ["A", "  ", ""]}')

    assert plan_subqueries("A?", llm_client) == ["A"]


def test_plan_subqueries_caps_at_max_subqueries():
    many = [f"q{i}" for i in range(MAX_SUBQUERIES + 3)]
    llm_client = StubLLM('{"subqueries": ' + str(many).replace("'", '"') + "}")

    result = plan_subqueries("many-part question", llm_client)

    assert result == many[:MAX_SUBQUERIES]


@pytest.mark.parametrize(
    "response",
    [
        "не json вообще",
        '{"subqueries": []}',
        '{"subqueries": ["", "   "]}',
        "{}",
        "",
    ],
)
def test_plan_subqueries_falls_back_to_original_question_on_bad_response(response):
    llm_client = StubLLM(response)

    result = plan_subqueries("Оригинальный вопрос", llm_client)

    assert result == ["Оригинальный вопрос"]


def test_plan_subqueries_falls_back_when_llm_call_raises():
    class FailingLLM:
        def generate(self, prompt: str) -> str:
            raise ValueError("LLM returned an empty response")

    result = plan_subqueries("Оригинальный вопрос", FailingLLM())

    assert result == ["Оригинальный вопрос"]


def test_build_planner_prompt_includes_question():
    prompt = build_planner_prompt("Сравни выручку за 2022 и 2023 год")

    assert "Сравни выручку за 2022 и 2023 год" in prompt
    assert "subqueries" in prompt
