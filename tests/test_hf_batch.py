import json
from pathlib import Path

import pytest
from datasets import Dataset

from file_agent.document import Block, Document
from file_agent.hf_batch import _create_cached_document_loader, generate_hf_qa_records
from file_agent.hf_rag import GeneratedQARecord, RetrievedContext


class DummyLLM:
    model = "fake/model"
    temperature = 0.0
    max_tokens = 128

    def generate(self, prompt: str) -> str:
        return "unused"


def make_dataset():
    return Dataset.from_dict(
        {
            "id": ["q0001", "q0002"],
            "question": ["First question?", "Second question?"],
            "answer": ["First gold answer", "Second gold answer"],
            "doc_ids": [["q0001/first.txt"], ["q0002/second.txt"]],
        }
    )


def make_generated_record(record, answer_suffix="generated"):
    metadata_json = json.dumps(
        {
            "dataset_doc_id": record.doc_ids[0],
            "dataset_record_id": record.id,
        },
        sort_keys=True,
    )
    return GeneratedQARecord(
        id=record.id,
        question=record.question,
        doc_ids=record.doc_ids,
        answer_model=f"{record.id} {answer_suffix}",
        contexts=(
            RetrievedContext(
                rank=1,
                chunk_id="block-1-chunk-1",
                document_id=record.doc_ids[0],
                text=f"Context for {record.id}",
                retrieval_text=f"Context for {record.id}",
                score=0.75,
                metadata_json=metadata_json,
            ),
        ),
        answer=record.answer,
    )


def install_fake_processor(monkeypatch, calls, fail_on_id=None):
    def fake_process_hf_qa_record(**kwargs):
        record = kwargs["record"]
        calls.append((record.id, kwargs))
        if record.id == fail_on_id:
            raise RuntimeError(f"Failed on {record.id}")
        return make_generated_record(record)

    monkeypatch.setattr(
        "file_agent.hf_batch.process_hf_qa_record",
        fake_process_hf_qa_record,
    )


def test_generate_hf_qa_records_processes_in_order_and_writes_checkpoints(
    monkeypatch,
    tmp_path,
):
    calls = []
    install_fake_processor(monkeypatch, calls)

    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        revision="commit-sha",
        top_k=3,
        max_chars=800,
        overlap=80,
    )

    assert [record.id for record in result.records] == ["q0001", "q0002"]
    assert result.processed_count == 2
    assert result.resumed_count == 0
    assert result.total_count == 2
    assert [record_id for record_id, _ in calls] == ["q0001", "q0002"]
    assert calls[0][1]["top_k"] == 3
    assert calls[0][1]["max_chars"] == 800
    assert calls[0][1]["overlap"] == 80
    assert calls[0][1]["document_loader"] is calls[1][1]["document_loader"]

    checkpoint_paths = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert [path.name for path in checkpoint_paths] == ["000000.json", "000001.json"]
    first_checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert first_checkpoint["schema_version"] == 3
    assert first_checkpoint["parameters"]["dataset_id"] == "owner/rag-qa"
    assert first_checkpoint["parameters"]["revision"] == "commit-sha"
    assert first_checkpoint["parameters"]["model_id"] == "fake/model"
    assert first_checkpoint["parameters"]["rag_pipeline_version"] == "section-token-small-to-big-v1"
    assert first_checkpoint["parameters"]["embedding_model"]
    assert first_checkpoint["parameters"]["top_k"] == 3
    assert first_checkpoint["result"] == result.records[0].to_dict()


def test_generate_hf_qa_records_passes_max_iterations_to_processor_and_checkpoint(
    monkeypatch,
    tmp_path,
):
    calls = []
    install_fake_processor(monkeypatch, calls)

    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        max_iterations=3,
    )

    assert calls[0][1]["max_iterations"] == 3
    checkpoint = json.loads((tmp_path / "checkpoints" / "000000.json").read_text(encoding="utf-8"))
    assert checkpoint["parameters"]["max_iterations"] == 3
    assert result.records[0].id == "q0001"


def test_generate_hf_qa_records_rejects_max_iterations_change_on_resume(
    monkeypatch,
    tmp_path,
):
    install_fake_processor(monkeypatch, [])
    generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        max_iterations=6,
    )

    with pytest.raises(ValueError, match="parameters do not match"):
        generate_hf_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
            max_iterations=3,
            resume=True,
        )


def test_cached_document_loader_reuses_parsing_and_returns_isolated_copies(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    document_path = tmp_path / "shared.txt"
    document_path.write_text("Shared document", encoding="utf-8")
    load_calls = []

    def fake_load_documents(file_paths):
        paths = list(file_paths)
        load_calls.append(paths)
        return [
            Document(
                file_name=paths[0].name,
                file_type="txt",
                blocks=[
                    Block(
                        id="block-1",
                        text="Shared document",
                        type="text",
                        metadata={"source_file": paths[0].name},
                    )
                ],
            )
        ]

    monkeypatch.setattr("file_agent.hf_batch.load_documents", fake_load_documents)
    loader = _create_cached_document_loader()

    relative_document_path = Path("shared.txt")
    first_document = loader([relative_document_path])[0]
    first_document.blocks[0].metadata["dataset_record_id"] = "q0001"
    second_document = loader([document_path])[0]

    assert load_calls == [[relative_document_path]]
    assert first_document is not second_document
    assert first_document.blocks[0] is not second_document.blocks[0]
    assert "dataset_record_id" not in second_document.blocks[0].metadata


def test_generate_hf_qa_records_resumes_without_processing_again(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls)
    first_run = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        revision="commit-sha",
    )

    def fail_if_called(**kwargs):
        raise AssertionError("Processor must not be called for completed rows")

    monkeypatch.setattr("file_agent.hf_batch.process_hf_qa_record", fail_if_called)
    resumed_run = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        revision="commit-sha",
        resume=True,
    )

    assert resumed_run.records == first_run.records
    assert resumed_run.processed_count == 0
    assert resumed_run.resumed_count == 2


def test_generate_hf_qa_records_continues_after_partial_failure(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls, fail_on_id="q0002")

    with pytest.raises(RuntimeError, match="Failed on q0002"):
        generate_hf_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
        )

    assert (tmp_path / "checkpoints" / "000000.json").exists()
    assert not (tmp_path / "checkpoints" / "000001.json").exists()

    resumed_calls = []
    install_fake_processor(monkeypatch, resumed_calls)
    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        resume=True,
    )

    assert [record_id for record_id, _ in resumed_calls] == ["q0002"]
    assert result.processed_count == 1
    assert result.resumed_count == 1
    assert [record.id for record in result.records] == ["q0001", "q0002"]


def test_generate_hf_qa_records_rejects_existing_checkpoints_without_resume(
    monkeypatch,
    tmp_path,
):
    install_fake_processor(monkeypatch, [])
    generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    with pytest.raises(FileExistsError, match="resume=True"):
        generate_hf_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
        )


def test_generate_hf_qa_records_rejects_parameter_changes_on_resume(
    monkeypatch,
    tmp_path,
):
    install_fake_processor(monkeypatch, [])
    generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        top_k=5,
    )

    with pytest.raises(ValueError, match="parameters do not match"):
        generate_hf_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
            top_k=3,
            resume=True,
        )
