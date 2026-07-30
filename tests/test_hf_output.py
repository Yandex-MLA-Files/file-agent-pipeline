import json
from dataclasses import replace

import pytest
from datasets import Dataset, load_from_disk

from file_agent.hf_output import (
    GENERATED_QA_FEATURES,
    build_generated_qa_dataset,
    save_generated_qa_dataset,
)
from file_agent.hf_rag import GeneratedQARecord, RetrievedContext


def make_source_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "id": ["q0001", "q0002"],
            "question": ["First question?", "Second question?"],
            "answer": ["First gold answer", "Second gold answer"],
            "doc_ids": [["q0001/first.txt"], ["q0002/second.txt"]],
        },
        split="train",
    )


def make_generated_records() -> tuple[GeneratedQARecord, ...]:
    return (
        GeneratedQARecord(
            id="q0001",
            question="First question?",
            doc_ids=("q0001/first.txt",),
            answer_model="First generated answer",
            contexts=(
                RetrievedContext(
                    rank=1,
                    chunk_id="block-1-chunk-1",
                    document_id="q0001/first.txt",
                    text="First retrieved context",
                    retrieval_text="First retrieved context",
                    score=0.75,
                    metadata_json=json.dumps({"page_number": 1}),
                ),
            ),
            answer="First gold answer",
        ),
        GeneratedQARecord(
            id="q0002",
            question="Second question?",
            doc_ids=("q0002/second.txt",),
            answer_model="Second generated answer",
            contexts=(),
            answer="Second gold answer",
        ),
    )


def test_build_generated_qa_dataset_uses_explicit_schema_and_source_order():
    dataset = build_generated_qa_dataset(make_source_dataset(), make_generated_records())

    assert dataset.column_names == [
        "id",
        "question",
        "doc_ids",
        "answer_model",
        "contexts",
        "answer",
    ]
    assert dataset.features == GENERATED_QA_FEATURES
    assert dataset.split == "train"
    assert dataset.num_rows == 2
    assert dataset[0]["contexts"] == [
        {
            "rank": 1,
            "chunk_id": "block-1-chunk-1",
            "document_id": "q0001/first.txt",
            "text": "First retrieved context",
            "retrieval_text": "First retrieved context",
            "score": 0.75,
            "metadata_json": '{"page_number": 1}',
        }
    ]
    assert dataset[1]["contexts"] == []


def test_build_generated_qa_dataset_rejects_missing_result():
    with pytest.raises(ValueError, match="count does not match"):
        build_generated_qa_dataset(make_source_dataset(), make_generated_records()[:1])


@pytest.mark.parametrize("field_name", ["id", "question", "doc_ids", "answer"])
def test_build_generated_qa_dataset_rejects_result_that_differs_from_source(
    field_name,
):
    records = list(make_generated_records())
    replacement_values = {
        "id": "another-id",
        "question": "Another question?",
        "doc_ids": ("q0001/another.txt",),
        "answer": "Another gold answer",
    }
    records[0] = replace(records[0], **{field_name: replacement_values[field_name]})

    with pytest.raises(ValueError, match=field_name):
        build_generated_qa_dataset(make_source_dataset(), records)


def test_save_generated_qa_dataset_writes_round_trippable_artifacts(tmp_path):
    source_dataset = make_source_dataset()
    records = make_generated_records()

    artifacts = save_generated_qa_dataset(source_dataset, records, tmp_path)

    assert artifacts.parquet_path == tmp_path / "answers.parquet"
    assert artifacts.hf_dataset_path == tmp_path / "hf_dataset"
    assert artifacts.row_count == 2
    assert artifacts.parquet_path.is_file()
    assert artifacts.hf_dataset_path.is_dir()

    parquet_dataset = Dataset.from_parquet(
        str(artifacts.parquet_path),
        cache_dir=str(tmp_path / "cache"),
    )
    disk_dataset = load_from_disk(artifacts.hf_dataset_path)
    expected_rows = [record.to_dict() for record in records]
    assert [parquet_dataset[index] for index in range(2)] == expected_rows
    assert [disk_dataset[index] for index in range(2)] == expected_rows
    assert disk_dataset.features == GENERATED_QA_FEATURES
    assert not list(tmp_path.glob(".*.tmp"))


def test_save_generated_qa_dataset_does_not_overwrite_existing_artifacts(tmp_path):
    parquet_path = tmp_path / "answers.parquet"
    parquet_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        save_generated_qa_dataset(make_source_dataset(), make_generated_records(), tmp_path)

    assert parquet_path.read_text(encoding="utf-8") == "keep me"
    assert not (tmp_path / "hf_dataset").exists()
