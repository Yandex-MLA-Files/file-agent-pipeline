from __future__ import annotations

import json

import pandas as pd
import pytest

from eval.resumable import (
    INCOMPLETE_ROWS_FILENAME,
    CheckpointError,
    IncompleteEvaluationError,
    evaluate_in_batches,
)


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


class _FakeJudge:
    metric_names = ("faithfulness", "answer_correctness")

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def evaluate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(run_df["id"].tolist())
        if self.fail_on_call == len(self.calls):
            raise ConnectionError("simulated connection failure")
        result = run_df.copy()
        result["faithfulness"] = 0.9
        result["answer_correctness"] = 0.8
        return result


def _write_run(tmp_path, n_rows: int = 5):
    run_df = _run_df(n_rows)
    run_path = tmp_path / "run.parquet"
    run_df.to_parquet(run_path, index=False)
    return run_df, run_path


def _evaluate(run_df, judge, run_path, out_dir, *, resume=False, batch_size=2):
    return evaluate_in_batches(
        run_df,
        judge,
        run_path=run_path,
        out_dir=out_dir,
        batch_size=batch_size,
        resume=resume,
        judge_model="deepseek-v4-flash",
        embedding_model="multilingual-minilm",
    )


def test_evaluates_in_batches_and_checkpoints_every_completed_row(tmp_path):
    run_df, run_path = _write_run(tmp_path)
    judge = _FakeJudge()

    result = _evaluate(run_df, judge, run_path, tmp_path / "out")

    assert judge.calls == [["ex_000", "ex_001"], ["ex_002", "ex_003"], ["ex_004"]]
    assert result.processed_rows == 5
    assert result.resumed_rows == 0
    assert result.batches == 3
    assert result.scored_df["id"].tolist() == run_df["id"].tolist()
    checkpoint = pd.read_parquet(tmp_path / "out" / "checkpoint.parquet")
    assert checkpoint["id"].tolist() == run_df["id"].tolist()


def test_resume_skips_batches_saved_before_a_failure(tmp_path):
    run_df, run_path = _write_run(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(ConnectionError, match="simulated"):
        _evaluate(run_df, _FakeJudge(fail_on_call=2), run_path, out_dir)

    checkpoint = pd.read_parquet(out_dir / "checkpoint.parquet")
    assert checkpoint["id"].tolist() == ["ex_000", "ex_001"]

    resumed_judge = _FakeJudge()
    result = _evaluate(run_df, resumed_judge, run_path, out_dir, resume=True)

    assert resumed_judge.calls == [["ex_002", "ex_003"], ["ex_004"]]
    assert result.resumed_rows == 2
    assert result.processed_rows == 3
    assert result.scored_df["id"].tolist() == run_df["id"].tolist()


def test_resume_rejects_a_different_judge_model(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=2)
    out_dir = tmp_path / "out"
    _evaluate(run_df, _FakeJudge(), run_path, out_dir)

    with pytest.raises(CheckpointError, match="judge_model"):
        evaluate_in_batches(
            run_df,
            _FakeJudge(),
            run_path=run_path,
            out_dir=out_dir,
            batch_size=2,
            resume=True,
            judge_model="another-model",
            embedding_model="multilingual-minilm",
        )


def test_resume_rejects_a_different_recorded_judge_config(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=2)
    out_dir = tmp_path / "out"
    common = {
        "run_path": run_path,
        "out_dir": out_dir,
        "batch_size": 2,
        "judge_model": "deepseek-v4-flash",
        "embedding_model": "multilingual-minilm",
    }
    evaluate_in_batches(
        run_df,
        _FakeJudge(),
        resume=False,
        judge_config={"max_tokens": 8192},
        **common,
    )

    with pytest.raises(CheckpointError, match="judge_config"):
        evaluate_in_batches(
            run_df,
            _FakeJudge(),
            resume=True,
            judge_config={"max_tokens": 4096},
            **common,
        )


def test_resume_upgrades_a_legacy_manifest_without_discarding_rows(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=3)
    out_dir = tmp_path / "out"
    _evaluate(run_df, _FakeJudge(), run_path, out_dir, batch_size=2)

    resumed_judge = _FakeJudge()
    result = evaluate_in_batches(
        run_df,
        resumed_judge,
        run_path=run_path,
        out_dir=out_dir,
        batch_size=1,
        resume=True,
        judge_model="deepseek-v4-flash",
        embedding_model="multilingual-minilm",
        judge_config={"max_tokens": 8192, "fallback_reasoning_mode": "DISABLED"},
    )

    assert result.resumed_rows == 3
    assert result.processed_rows == 0
    assert resumed_judge.calls == []
    manifest = json.loads((out_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_config"] == {
        "max_tokens": 8192,
        "fallback_reasoning_mode": "DISABLED",
    }
    assert manifest["legacy_completed_rows"] == 3


def test_incomplete_scores_are_not_marked_as_completed(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=2)

    class _IncompleteJudge(_FakeJudge):
        def evaluate(self, batch_df: pd.DataFrame) -> pd.DataFrame:
            result = super().evaluate(batch_df)
            result.loc[result.index[-1], "faithfulness"] = float("nan")
            return result

    with pytest.raises(IncompleteEvaluationError, match="retry only"):
        _evaluate(run_df, _IncompleteJudge(), run_path, tmp_path / "out")

    checkpoint = pd.read_parquet(tmp_path / "out" / "checkpoint.parquet")
    assert checkpoint["id"].tolist() == ["ex_000"]
    incomplete = json.loads(
        (tmp_path / "out" / INCOMPLETE_ROWS_FILENAME).read_text(encoding="utf-8")
    )
    assert incomplete["rows"] == [{"id": "ex_001", "missing_metrics": ["faithfulness"]}]


def test_incomplete_row_does_not_block_later_rows_and_resume_retries_only_it(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=4)
    out_dir = tmp_path / "out"

    class _OneIncompleteJudge(_FakeJudge):
        def evaluate(self, batch_df: pd.DataFrame) -> pd.DataFrame:
            result = super().evaluate(batch_df)
            result.loc[result["id"] == "ex_001", "faithfulness"] = float("nan")
            return result

    first_judge = _OneIncompleteJudge()
    with pytest.raises(IncompleteEvaluationError, match="ex_001"):
        _evaluate(run_df, first_judge, run_path, out_dir, batch_size=2)

    assert first_judge.calls == [["ex_000", "ex_001"], ["ex_002", "ex_003"]]
    checkpoint = pd.read_parquet(out_dir / "checkpoint.parquet")
    assert checkpoint["id"].tolist() == ["ex_000", "ex_002", "ex_003"]

    resumed_judge = _FakeJudge()
    result = _evaluate(
        run_df,
        resumed_judge,
        run_path,
        out_dir,
        resume=True,
        batch_size=2,
    )

    assert resumed_judge.calls == [["ex_001"]]
    assert result.resumed_rows == 3
    assert result.processed_rows == 1
    assert result.scored_df["id"].tolist() == run_df["id"].tolist()
    assert not (out_dir / INCOMPLETE_ROWS_FILENAME).exists()


def test_batch_size_must_be_positive(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=1)

    with pytest.raises(ValueError, match="greater than zero"):
        _evaluate(run_df, _FakeJudge(), run_path, tmp_path / "out", batch_size=0)
