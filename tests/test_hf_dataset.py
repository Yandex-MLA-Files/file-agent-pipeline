import pytest
from datasets import Dataset

from file_agent.hf_dataset import (
    QADatasetRecord,
    download_record_documents,
    load_qa_dataset,
    validate_qa_dataset,
)


def make_dataset(**overrides):
    data = {
        "id": ["q0001", "q0002"],
        "question": ["Первый вопрос?", "Второй вопрос?"],
        "answer": ["Первый ответ", "Второй ответ"],
        "doc_ids": [
            ["q0001/Документ один.pdf"],
            ["q0002/notes.txt", "q0002/table.docx"],
        ],
    }
    data.update(overrides)
    return Dataset.from_dict(data)


def test_load_qa_dataset_passes_hugging_face_options(monkeypatch, tmp_path):
    expected_dataset = make_dataset()
    calls = []

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return expected_dataset

    monkeypatch.setattr("file_agent.hf_dataset.load_dataset", fake_load_dataset)

    dataset = load_qa_dataset(
        dataset_id="owner/rag-qa",
        config_name="default",
        split="train",
        revision="commit-sha",
        cache_dir=tmp_path,
        token="test-token",
    )

    assert dataset is expected_dataset
    assert calls == [
        {
            "path": "owner/rag-qa",
            "name": "default",
            "split": "train",
            "revision": "commit-sha",
            "cache_dir": str(tmp_path),
            "token": "test-token",
        }
    ]


def test_validate_qa_dataset_rejects_missing_columns():
    dataset = Dataset.from_dict(
        {
            "id": ["q0001"],
            "question": ["Question"],
        }
    )

    with pytest.raises(ValueError, match="answer, doc_ids"):
        validate_qa_dataset(dataset)


def test_validate_qa_dataset_rejects_duplicate_ids():
    dataset = make_dataset(id=["q0001", "q0001"])

    with pytest.raises(ValueError, match="Duplicate id at row 1: q0001"):
        validate_qa_dataset(dataset)


def test_qa_dataset_record_preserves_doc_id_order():
    row = make_dataset()[1]

    record = QADatasetRecord.from_row(row)

    assert record == QADatasetRecord(
        id="q0002",
        question="Второй вопрос?",
        answer="Второй ответ",
        doc_ids=("q0002/notes.txt", "q0002/table.docx"),
    )


def test_download_record_documents_uses_exact_doc_ids(monkeypatch, tmp_path):
    record = QADatasetRecord.from_row(make_dataset()[1])
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        file_path = tmp_path / f"download-{len(calls)}"
        file_path.touch()
        return str(file_path)

    monkeypatch.setattr("file_agent.hf_dataset.hf_hub_download", fake_hf_hub_download)

    paths = download_record_documents(
        record=record,
        dataset_id="owner/rag-qa",
        revision="commit-sha",
        cache_dir=tmp_path / "cache",
        token="test-token",
    )

    assert paths == [tmp_path / "download-1", tmp_path / "download-2"]
    assert calls == [
        {
            "repo_id": "owner/rag-qa",
            "repo_type": "dataset",
            "filename": "q0002/notes.txt",
            "revision": "commit-sha",
            "cache_dir": str(tmp_path / "cache"),
            "token": "test-token",
        },
        {
            "repo_id": "owner/rag-qa",
            "repo_type": "dataset",
            "filename": "q0002/table.docx",
            "revision": "commit-sha",
            "cache_dir": str(tmp_path / "cache"),
            "token": "test-token",
        },
    ]


@pytest.mark.parametrize(
    "doc_id",
    [
        "../secret.txt",
        "q0001/../secret.txt",
        "/absolute/file.txt",
        "q0001\\file.txt",
        "q0001//file.txt",
    ],
)
def test_qa_dataset_record_rejects_unsafe_doc_ids(doc_id):
    row = {
        "id": "q0001",
        "question": "Question",
        "answer": "Answer",
        "doc_ids": [doc_id],
    }

    with pytest.raises(ValueError, match="doc_id"):
        QADatasetRecord.from_row(row)
