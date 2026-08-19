from pathlib import Path

import pytest

from file_agent.query_expansion import (
    MultiQueryExpander,
    MultiQuerySettings,
    parse_variants,
    resolve_query_expander,
)


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.model = "scripted"

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("no reply scripted")
        return self.replies.pop(0)


def test_parse_variants_strips_numbering_quotes_and_chatter():
    raw = (
        "Here are three formulations:\n"
        '1. "Какова выручка компании за год?"\n'
        "- Сколько компания заработала за отчётный период\n"
        "• Годовой доход организации\n"
        "Сколько компания заработала за отчётный период\n"  # duplicate
        "Какая выручка у компании?\n"  # over the limit
    )
    variants = parse_variants(raw, "Какая выручка у компании?", limit=3)
    assert variants == [
        "Какова выручка компании за год?",
        "Сколько компания заработала за отчётный период",
        "Годовой доход организации",
    ]


def test_parse_variants_drops_the_original_question():
    raw = "какая выручка у компании\nДоход компании"
    assert parse_variants(raw, "Какая выручка у компании?", limit=3) == ["Доход компании"]


def test_paraphrase_mode_asks_once_and_caches(tmp_path):
    llm = ScriptedLLM(["Доход компании\nСколько заработала компания"])
    expander = MultiQueryExpander(llm, mode="paraphrase", count=2, cache_dir=tmp_path)

    first = expander("Какая выручка у компании?")
    second = expander("Какая выручка у компании?")

    assert first == ["Доход компании", "Сколько заработала компания"]
    assert second == first
    assert len(llm.prompts) == 1  # the second call came from the cache
    assert list(tmp_path.rglob("*.json"))


def test_mixed_mode_appends_a_hypothetical_passage(tmp_path):
    llm = ScriptedLLM(
        [
            "Доход компании\nСколько заработала компания",
            "Выручка компании за отчётный год составила 12,5 млрд рублей, "
            "что на 8 % выше прошлогодней.",
        ]
    )
    expander = MultiQueryExpander(llm, mode="mixed", count=3, cache_dir=None)

    variants = expander("Какая выручка у компании?")

    assert len(variants) == 3
    assert variants[-1].startswith("Выручка компании за отчётный год")
    assert "Write 2 alternative formulations" in llm.prompts[0]
    assert "hypothetical" not in llm.prompts[1].lower() or "passage" in llm.prompts[1]


def test_hyde_mode_returns_only_the_passage(tmp_path):
    llm = ScriptedLLM(["Пассаж в стиле документа."])
    expander = MultiQueryExpander(llm, mode="hyde", count=1, cache_dir=None)

    assert expander("Вопрос?") == ["Пассаж в стиле документа."]
    assert len(llm.prompts) == 1


def test_model_failure_yields_no_variants_instead_of_raising():
    llm = ScriptedLLM([])  # every call raises
    expander = MultiQueryExpander(llm, mode="paraphrase", count=3, cache_dir=None)

    assert expander("Вопрос?") == []


def test_settings_from_env_validate_mode_and_count(monkeypatch):
    monkeypatch.setenv("MULTI_QUERY", "on")
    monkeypatch.setenv("MULTI_QUERY_MODE", "mixed")
    monkeypatch.setenv("MULTI_QUERY_COUNT", "4")
    monkeypatch.setenv("MULTI_QUERY_CACHE", "off")
    settings = MultiQuerySettings.from_env()
    assert (settings.enabled, settings.mode, settings.count, settings.cache_dir) == (
        True,
        "mixed",
        4,
        None,
    )

    monkeypatch.setenv("MULTI_QUERY_MODE", "magic")
    with pytest.raises(ValueError):
        MultiQuerySettings.from_env()

    monkeypatch.setenv("MULTI_QUERY_MODE", "paraphrase")
    monkeypatch.setenv("MULTI_QUERY_COUNT", "0")
    with pytest.raises(ValueError):
        MultiQuerySettings.from_env()


def test_expander_is_off_unless_enabled(monkeypatch):
    monkeypatch.delenv("MULTI_QUERY", raising=False)
    assert resolve_query_expander() is None
    monkeypatch.setenv("MULTI_QUERY", "false")
    assert resolve_query_expander() is None


def test_cache_key_includes_the_mode_and_count(tmp_path):
    llm = ScriptedLLM(["A\nB", "C"])
    two = MultiQueryExpander(llm, mode="paraphrase", count=2, cache_dir=tmp_path)
    one = MultiQueryExpander(llm, mode="paraphrase", count=1, cache_dir=tmp_path)

    assert two("Q?") == ["A", "B"]
    assert one("Q?") == ["C"]
    assert len({p.name for p in Path(tmp_path).rglob("*.json")}) == 2
