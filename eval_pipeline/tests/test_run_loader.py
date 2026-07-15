import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_loader import RunValidationError, load_run


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "ex_001",
                "question": "question one?",
                "answer": "answer one",
                "contexts": ["chunk 1", "chunk 2"],
                "ground_truth": "reference one",
            },
            {
                "id": "ex_002",
                "question": "question two?",
                "answer": "answer two",
                "contexts": ["chunk 3"],
                "ground_truth": "reference two",
            },
        ]
    )


def test_valid_run_loads(tmp_path):
    path = tmp_path / "run.parquet"
    _valid_df().to_parquet(path, index=False)
    df = load_run(path)
    assert len(df) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(RunValidationError, match="file not found"):
        load_run(tmp_path / "nope.parquet")


def test_missing_column_raises(tmp_path):
    df = _valid_df().drop(columns=["ground_truth"])
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="missing required columns"):
        load_run(path)


def test_duplicate_id_raises(tmp_path):
    df = pd.concat([_valid_df(), _valid_df().iloc[[0]]], ignore_index=True)
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="duplicate ids"):
        load_run(path)


def test_contexts_as_string_raises(tmp_path):
    df = _valid_df()
    # both rows — pyarrow rejects a mixed-type column otherwise
    df["contexts"] = "joined string instead of a list"
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="list\\[str\\]"):
        load_run(path)


def test_empty_answer_raises(tmp_path):
    df = _valid_df()
    df.loc[0, "answer"] = ""
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="empty answer"):
        load_run(path)
