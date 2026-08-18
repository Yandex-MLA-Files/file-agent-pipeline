import json
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
                "answer_model": "answer one",
                "contexts": ["chunk 1", "chunk 2"],
                "answer": "reference one",
            },
            {
                "id": "ex_002",
                "question": "question two?",
                "answer_model": "answer two",
                "contexts": ["chunk 3"],
                "answer": "reference two",
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
    df = _valid_df().drop(columns=["answer"])
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


def test_contexts_as_chunk_dicts_extracts_text(tmp_path):
    df = _valid_df()
    df["contexts"] = [
        [{"rank": 1, "chunk_id": "c1", "text": "chunk 1", "score": 0.9}],
        [{"rank": 1, "chunk_id": "c2", "text": "chunk 3", "score": 0.5}],
    ]
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    loaded = load_run(path)
    assert loaded.loc[0, "contexts"] == ["chunk 1"]
    assert loaded.loc[1, "contexts"] == ["chunk 3"]


def test_contexts_as_chunk_dicts_preserve_source_and_pages(tmp_path):
    df = _valid_df().iloc[[0]].copy()
    df["contexts"] = [
        [
            {
                "rank": 1,
                "document_id": "q0001/document.pdf",
                "text": "chunk text",
                "metadata_json": json.dumps(
                    {
                        "source_file": "document.pdf",
                        "page_number": 7,
                        "page_numbers": [7, 8],
                    }
                ),
            }
        ]
    ]
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)

    loaded = load_run(path)

    assert loaded.loc[0, "contexts"] == ["[source: document.pdf; pages: 7, 8]\nchunk text"]


def test_context_metadata_falls_back_to_document_id(tmp_path):
    df = _valid_df().iloc[[0]].copy()
    df["contexts"] = [
        [
            {
                "document_id": "q0001/document.pdf",
                "text": "chunk text",
                "metadata_json": "not-json",
            }
        ]
    ]
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)

    loaded = load_run(path)

    assert loaded.loc[0, "contexts"] == ["[source: q0001/document.pdf]\nchunk text"]


def test_contexts_chunk_dict_without_text_raises(tmp_path):
    df = _valid_df()
    df["contexts"] = [
        [{"rank": 1, "chunk_id": "c1", "score": 0.9}],
        [{"rank": 1, "chunk_id": "c2", "text": "chunk 3", "score": 0.5}],
    ]
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="'text' key"):
        load_run(path)


def test_empty_answer_model_raises(tmp_path):
    df = _valid_df()
    df.loc[0, "answer_model"] = ""
    path = tmp_path / "run.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RunValidationError, match="empty answer_model"):
        load_run(path)
