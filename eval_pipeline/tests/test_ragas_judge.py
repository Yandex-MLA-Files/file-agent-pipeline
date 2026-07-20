import json
import re

import pandas as pd
import pytest
from ragas.metrics import (
    AspectCritic,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from eval.judge.ragas_judge import RagasJudge, _TokenUsageCallback, _usage_cost

# score = row index (from user_input's trailing digit) + a per-metric offset,
# so each (row, metric) cell is independently verifiable.
_METRIC_OFFSETS = {
    "faithfulness": 0.01,
    "answer_correctness": 0.02,
    "answer_relevancy": 0.03,
    "context_precision": 0.04,
    "context_recall": 0.05,
}


def _row_index(sample) -> int | None:

    for value in (sample.user_input, sample.response, sample.reference):
        if value:
            match = re.search(r"\d+", value)
            if match:
                return int(match.group())
    return None


def _score_for(name: str, sample) -> float:
    row_index = _row_index(sample)
    if row_index is None:
        return 0.0
    return row_index + _METRIC_OFFSETS[name]


@pytest.fixture
def patched_metrics(monkeypatch):
    for cls in (
        Faithfulness,
        FactualCorrectness,
        AspectCritic,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    ):

        async def fake_ascore(self, sample, callbacks, _cls=cls):
            return _score_for(self.name, sample)

        monkeypatch.setattr(cls, "_single_turn_ascore", fake_ascore)


def _run_df(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"ex_{i:03d}",
                "question": f"q{i}",
                "answer_model": f"a{i}",
                "contexts": [f"c{i}"],
                "answer": f"gt{i}",
            }
            for i in range(n_rows)
        ]
    )


def test_evaluate_returns_one_row_per_input_row(patched_metrics):
    judge = RagasJudge(model="test-model", llm=object())
    scored = judge.evaluate(_run_df(3))

    assert len(scored) == 3
    assert list(scored["id"]) == ["ex_000", "ex_001", "ex_002"]


def test_evaluate_adds_all_five_metric_columns_with_correct_values(patched_metrics):
    judge = RagasJudge(model="test-model", llm=object())
    scored = judge.evaluate(_run_df(2))

    for name, offset in _METRIC_OFFSETS.items():
        assert scored.loc[0, name] == pytest.approx(0 + offset)
        assert scored.loc[1, name] == pytest.approx(1 + offset)


def test_evaluate_preserves_original_run_columns(patched_metrics):
    judge = RagasJudge(model="test-model", llm=object())
    run_df = _run_df(1)
    scored = judge.evaluate(run_df)

    for col in ("id", "question", "answer_model", "contexts", "answer"):
        assert col in scored.columns
        assert scored.loc[0, col] == run_df.loc[0, col]


def test_metric_names_match_report_expectations():
    assert RagasJudge.metric_names == (
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


def _is_russian(text: str) -> bool:
    return any("а" <= c <= "я" or "А" <= c <= "Я" for c in text)


def test_prompts_are_localized_to_russian_not_ragas_defaults():

    judge = RagasJudge(model="test-model", llm=object())
    faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall = (
        judge._metrics
    )

    assert _is_russian(faithfulness.statement_generator_prompt.instruction)
    assert _is_russian(faithfulness.statement_generator_prompt.examples[0][0].question)
    assert _is_russian(faithfulness.nli_statements_prompt.instruction)
    assert _is_russian(faithfulness.nli_statements_prompt.examples[0][0].context)

    assert _is_russian(answer_correctness.nli_prompt.instruction)
    assert _is_russian(answer_correctness.claim_decomposition_prompt.instruction)
    assert _is_russian(answer_correctness.claim_decomposition_prompt.examples[0][0].response)

    assert _is_russian(answer_relevancy.single_turn_prompt.instruction)
    assert _is_russian(answer_relevancy.definition)

    assert _is_russian(context_precision.context_precision_prompt.instruction)
    assert _is_russian(context_precision.context_precision_prompt.examples[0][0].question)

    assert _is_russian(context_recall.context_recall_prompt.instruction)
    assert _is_russian(context_recall.context_recall_prompt.examples[0][0].question)


class _FakeLLMResult:
    def __init__(self, llm_output):
        self.llm_output = llm_output


def test_token_usage_callback_accumulates_across_calls():
    cb = _TokenUsageCallback()
    cb.on_llm_end(
        _FakeLLMResult(
            {
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 30},
                },
                "model_name": "gpt://folder/deepseek-v4-flash/latest",
            }
        )
    )
    cb.on_llm_end(_FakeLLMResult({"token_usage": {"prompt_tokens": 50, "completion_tokens": 10}}))

    assert cb.input_tokens == 150
    assert cb.output_tokens == 30
    assert cb.cached_tokens == 30
    assert cb.model == "gpt://folder/deepseek-v4-flash/latest"


def test_token_usage_callback_handles_missing_llm_output():
    cb = _TokenUsageCallback()
    cb.on_llm_end(_FakeLLMResult(None))

    assert cb.input_tokens == 0
    assert cb.output_tokens == 0
    assert cb.cached_tokens == 0


def test_usage_cost_bills_cached_tokens_at_the_cached_rate(monkeypatch):
    monkeypatch.setenv("JUDGE_PRICE_PER_1K_INPUT_TOKENS", "0.3")
    monkeypatch.setenv("JUDGE_PRICE_PER_1K_OUTPUT_TOKENS", "0.5")
    monkeypatch.setenv("JUDGE_PRICE_PER_1K_CACHED_TOKENS", "0.075")

    cost = _usage_cost(input_tokens=1000, output_tokens=1000, cached_tokens=200)

    # 800 input tokens billed at the input rate, 200 at the cached rate,
    # 1000 output tokens at the output rate.
    expected = 800 / 1000 * 0.3 + 200 / 1000 * 0.075 + 1000 / 1000 * 0.5
    assert cost == pytest.approx(expected)


def test_evaluate_appends_a_usage_log_entry(patched_metrics, tmp_path):
    judge = RagasJudge(model="test-model", llm=object())

    judge.evaluate(_run_df(2))

    lines = (tmp_path / "usage_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["n_rows"] == 2
    assert entry["model"] == "test-model"
    # patched_metrics monkeypatches _single_turn_ascore directly, bypassing
    # the real LLM call -- on_llm_end never fires here, so this only checks
    # the log entry gets written with the right shape, not real token counts.
    assert entry["input_tokens"] == 0
    assert entry["output_tokens"] == 0
    assert entry["cached_tokens"] == 0
    assert entry["cost_rub"] == 0
    assert "timestamp" in entry
