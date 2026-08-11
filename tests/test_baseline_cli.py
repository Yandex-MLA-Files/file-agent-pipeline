import json

import pytest
from datasets import Dataset

from file_agent.baseline_cli import (
    BaselineGenerationConfig,
    BatchGenerationResult,
    create_argument_parser,
    generate_baseline_qa_records,
    main,
    process_baseline_hf_qa_record,
    process_baseline_qa_record,
    run_baseline_generation,
)
from file_agent.hf_dataset import QADatasetRecord
from file_agent.hf_output import GeneratedDatasetArtifacts
from file_agent.hf_rag import GeneratedQARecord, RetrievedContext
from file_agent.retrieval import SearchResult


class DummyLLM:
    model = "fake/model"
    temperature = 0.0
    max_tokens = 128

    def __init__(self, answer: str = "Generated answer"):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer

    def generate_with_tools(self, messages, tools, tool_choice="auto"):
        raise AssertionError("the plain RAG baseline must not use tool-calling")


class FakeRetriever:
    def __init__(self):
        self.chunks = []
        self.index_calls = 0
        self.search_calls = []
        self.clear_calls = 0

    def index(self, chunks):
        self.index_calls += 1
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return [
            SearchResult(chunk=chunk, score=1.0 / rank)
            for rank, chunk in enumerate(self.chunks[:top_k], start=1)
        ]

    def clear(self):
        self.clear_calls += 1


def make_record():
    return QADatasetRecord(
        id="q0001",
        question="Which contexts were found?",
        answer="Gold answer",
        doc_ids=("q0001/first.txt", "q0001/second.txt"),
    )


def create_text_documents(tmp_path):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("First retrieved context", encoding="utf-8")
    second_path.write_text("Second retrieved context", encoding="utf-8")
    return [first_path, second_path]


def test_process_baseline_qa_record_does_one_search_and_one_generate_call(tmp_path):
    record = make_record()
    document_paths = create_text_documents(tmp_path)
    llm_client = DummyLLM("Generated answer")
    retriever = FakeRetriever()

    result = process_baseline_qa_record(
        record=record,
        document_paths=document_paths,
        llm_client=llm_client,
        top_k=2,
        retriever=retriever,
    )

    assert result.id == "q0001"
    assert result.answer_model == "Generated answer"
    assert result.answer == "Gold answer"
    assert retriever.index_calls == 1
    assert retriever.search_calls == [(record.question, 2)]
    assert retriever.clear_calls == 1
    assert len(llm_client.prompts) == 1

    first_context, second_context = result.contexts
    assert first_context.document_id == "q0001/first.txt"
    assert first_context.text == "First retrieved context"
    assert second_context.document_id == "q0001/second.txt"


def test_process_baseline_qa_record_rejects_mismatched_document_paths(tmp_path):
    document_paths = create_text_documents(tmp_path)

    with pytest.raises(ValueError, match="document_paths count"):
        process_baseline_qa_record(
            record=make_record(),
            document_paths=document_paths[:1],
            llm_client=DummyLLM(),
            retriever=FakeRetriever(),
        )


def test_process_baseline_qa_record_clears_retriever_on_failure(tmp_path):
    class FailingLLM(DummyLLM):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("Generation failed")

    retriever = FakeRetriever()

    with pytest.raises(RuntimeError, match="Generation failed"):
        process_baseline_qa_record(
            record=make_record(),
            document_paths=create_text_documents(tmp_path),
            llm_client=FailingLLM(),
            retriever=retriever,
        )

    assert retriever.clear_calls == 1


def test_process_baseline_hf_qa_record_downloads_documents_before_processing(monkeypatch, tmp_path):
    record = make_record()
    document_paths = create_text_documents(tmp_path)
    calls = []

    def fake_download_record_documents(**kwargs):
        calls.append(kwargs)
        return document_paths

    monkeypatch.setattr(
        "file_agent.baseline_cli.download_record_documents",
        fake_download_record_documents,
    )

    result = process_baseline_hf_qa_record(
        record=record,
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        revision="commit-sha",
        cache_dir=tmp_path / "cache",
        token="test-token",
        retriever=FakeRetriever(),
    )

    assert result.id == record.id
    assert calls == [
        {
            "record": record,
            "dataset_id": "owner/rag-qa",
            "revision": "commit-sha",
            "cache_dir": tmp_path / "cache",
            "token": "test-token",
        }
    ]


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
    metadata_json = json.dumps({"dataset_doc_id": record.doc_ids[0]}, sort_keys=True)
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
    def fake_process(**kwargs):
        record = kwargs["record"]
        calls.append((record.id, kwargs))
        if record.id == fail_on_id:
            raise RuntimeError(f"Failed on {record.id}")
        return make_generated_record(record)

    monkeypatch.setattr("file_agent.baseline_cli.process_baseline_hf_qa_record", fake_process)


def test_generate_baseline_qa_records_processes_in_order_and_writes_checkpoints(
    monkeypatch, tmp_path
):
    calls = []
    install_fake_processor(monkeypatch, calls)

    result = generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        revision="commit-sha",
        top_k=3,
    )

    assert [record.id for record in result.records] == ["q0001", "q0002"]
    assert result.processed_count == 2
    assert result.resumed_count == 0

    checkpoint_paths = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert [path.name for path in checkpoint_paths] == ["000000.json", "000001.json"]
    first_checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert first_checkpoint["schema_version"] == 1
    assert first_checkpoint["parameters"]["pipeline_kind"] == "plain_rag_baseline"
    assert first_checkpoint["parameters"]["dataset_id"] == "owner/rag-qa"
    assert first_checkpoint["parameters"]["model_id"] == "fake/model"
    assert "agent_pipeline_version" not in first_checkpoint["parameters"]
    assert "tool_names" not in first_checkpoint["parameters"]


def test_generate_baseline_qa_records_resumes_without_processing_again(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls)
    first_run = generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("Processor must not be called for completed rows")

    monkeypatch.setattr("file_agent.baseline_cli.process_baseline_hf_qa_record", fail_if_called)
    resumed_run = generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        resume=True,
    )

    assert resumed_run.records == first_run.records
    assert resumed_run.processed_count == 0
    assert resumed_run.resumed_count == 2


def test_generate_baseline_qa_records_continues_after_partial_failure(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls, fail_on_id="q0002")

    with pytest.raises(RuntimeError, match="Failed on q0002"):
        generate_baseline_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
        )

    assert (tmp_path / "checkpoints" / "000000.json").exists()
    assert not (tmp_path / "checkpoints" / "000001.json").exists()

    resumed_calls = []
    install_fake_processor(monkeypatch, resumed_calls)
    result = generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        resume=True,
    )

    assert [record_id for record_id, _ in resumed_calls] == ["q0002"]
    assert result.processed_count == 1
    assert result.resumed_count == 1


def test_generate_baseline_qa_records_rejects_existing_checkpoints_without_resume(
    monkeypatch, tmp_path
):
    install_fake_processor(monkeypatch, [])
    generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    with pytest.raises(FileExistsError, match="resume=True"):
        generate_baseline_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
        )


def test_generate_baseline_qa_records_rejects_parameter_changes_on_resume(monkeypatch, tmp_path):
    install_fake_processor(monkeypatch, [])
    generate_baseline_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        top_k=5,
    )

    with pytest.raises(ValueError, match="parameters do not match"):
        generate_baseline_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
            top_k=3,
            resume=True,
        )


def test_run_baseline_generation_orchestrates_limited_run_and_writes_manifest(
    monkeypatch, tmp_path
):
    source_dataset = make_dataset()
    llm_client = DummyLLM()
    calls = {}

    def fake_create_llm_client(**kwargs):
        calls["llm"] = kwargs
        return llm_client

    def fake_load_qa_dataset(**kwargs):
        calls["load"] = kwargs
        return source_dataset

    def fake_generate_baseline_qa_records(**kwargs):
        calls["generate"] = kwargs
        selected_dataset = kwargs["dataset"]
        records = tuple(
            make_generated_record(QADatasetRecord.from_row(row, row_index=index))
            for index, row in enumerate(selected_dataset)
        )
        return BatchGenerationResult(records=records, processed_count=1, resumed_count=0)

    def fake_save_generated_qa_dataset(**kwargs):
        calls["save"] = kwargs
        output_dir = tmp_path / "run"
        parquet_path = output_dir / "answers.parquet"
        hf_dataset_path = output_dir / "hf_dataset"
        output_dir.mkdir(parents=True)
        parquet_path.write_text("parquet", encoding="utf-8")
        hf_dataset_path.mkdir()
        return GeneratedDatasetArtifacts(
            parquet_path=parquet_path,
            hf_dataset_path=hf_dataset_path,
            row_count=1,
        )

    monkeypatch.setattr(
        "file_agent.baseline_cli.create_generation_llm_client", fake_create_llm_client
    )
    monkeypatch.setattr("file_agent.baseline_cli.load_qa_dataset", fake_load_qa_dataset)
    monkeypatch.setattr(
        "file_agent.baseline_cli.generate_baseline_qa_records",
        fake_generate_baseline_qa_records,
    )
    monkeypatch.setattr(
        "file_agent.baseline_cli.save_generated_qa_dataset",
        fake_save_generated_qa_dataset,
    )
    monkeypatch.setattr("file_agent.baseline_cli._utc_timestamp", lambda: "2026-07-19T10:00:00Z")

    config = BaselineGenerationConfig(
        dataset_id="owner/rag-qa",
        output_dir=tmp_path / "run",
        top_k=3,
        limit=1,
        resume=True,
    )
    result = run_baseline_generation(config)

    assert calls["generate"]["resume"] is True
    assert calls["save"]["records"] == result.batch.records

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["generation"]["pipeline_kind"] == "plain_rag_baseline"
    assert manifest["generation"]["model_id"] == "fake/model"
    assert manifest["result"] == {"total_count": 1, "processed_count": 1, "resumed_count": 0}


def test_main_maps_cli_arguments(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "run"

    class FakeResult:
        def __init__(self):
            self.batch = BatchGenerationResult(records=(), processed_count=0, resumed_count=0)
            self.artifacts = GeneratedDatasetArtifacts(
                parquet_path=output_dir / "answers.parquet",
                hf_dataset_path=output_dir / "hf_dataset",
                row_count=0,
            )
            self.manifest_path = output_dir / "run_manifest.json"

    calls = []

    def fake_run(config):
        calls.append(config)
        return FakeResult()

    monkeypatch.setattr("file_agent.baseline_cli.run_baseline_generation", fake_run)
    monkeypatch.setattr("file_agent.baseline_cli.configure_telemetry", lambda: None)

    exit_code = main(
        [
            "--dataset-id",
            "owner/rag-qa",
            "--output-dir",
            str(output_dir),
            "--limit",
            "5",
            "--resume",
        ]
    )

    assert exit_code == 0
    assert calls[0].dataset_id == "owner/rag-qa"
    assert calls[0].limit == 5
    assert calls[0].resume is True
    output = capsys.readouterr().out
    assert "Completed 0 rows" in output


def test_create_argument_parser_requires_dataset_id_and_output_dir():
    parser = create_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
