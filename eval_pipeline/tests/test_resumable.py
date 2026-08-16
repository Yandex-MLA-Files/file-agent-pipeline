from __future__ import annotations

import pandas as pd
import pytest

from eval.resumable import CheckpointError, evaluate_in_batches


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


def test_incomplete_scores_are_not_marked_as_completed(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=2)

    class _IncompleteJudge(_FakeJudge):
        def evaluate(self, batch_df: pd.DataFrame) -> pd.DataFrame:
            result = super().evaluate(batch_df)
            result.loc[result.index[-1], "faithfulness"] = float("nan")
            return result

    with pytest.raises(RuntimeError, match="must be retried"):
        _evaluate(run_df, _IncompleteJudge(), run_path, tmp_path / "out")

    checkpoint = pd.read_parquet(tmp_path / "out" / "checkpoint.parquet")
    assert checkpoint["id"].tolist() == ["ex_000"]


def test_batch_size_must_be_positive(tmp_path):
    run_df, run_path = _write_run(tmp_path, n_rows=1)

    with pytest.raises(ValueError, match="greater than zero"):
        _evaluate(run_df, _FakeJudge(), run_path, tmp_path / "out", batch_size=0)
