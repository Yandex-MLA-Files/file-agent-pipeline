import json
import time
from pathlib import Path

import pytest
from datasets import Dataset

from file_agent.document import Block, Document
from file_agent.hf_batch import (
    _create_cached_document_loader,
    duration_stats,
    generate_hf_qa_records,
)
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


def test_generate_hf_qa_records_processes_every_row_and_writes_checkpoints(
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

    # Output order always matches the source dataset, even though rows
    # process concurrently and can finish in any order (see
    # test_generate_hf_qa_records_output_order_survives_out_of_order_completion).
    assert [record.id for record in result.records] == ["q0001", "q0002"]
    assert result.processed_count == 2
    assert result.resumed_count == 0
    assert result.total_count == 2
    assert {record_id for record_id, _ in calls} == {"q0001", "q0002"}
    calls_by_id = {record_id: kwargs for record_id, kwargs in calls}
    assert calls_by_id["q0001"]["top_k"] == 3
    assert calls_by_id["q0001"]["max_chars"] == 800
    assert calls_by_id["q0001"]["overlap"] == 80
    assert calls_by_id["q0001"]["document_loader"] is calls_by_id["q0002"]["document_loader"]

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


def test_generate_hf_qa_records_records_per_row_durations(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls)

    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    assert len(result.durations_seconds) == 2
    assert all(duration is not None and duration >= 0 for duration in result.durations_seconds)

    checkpoint = json.loads((tmp_path / "checkpoints" / "000000.json").read_text(encoding="utf-8"))
    assert checkpoint["duration_seconds"] == pytest.approx(result.durations_seconds[0])


def test_generate_hf_qa_records_carries_duration_forward_on_resume(monkeypatch, tmp_path):
    install_fake_processor(monkeypatch, [])
    first_run = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("Processor must not be called for completed rows")

    monkeypatch.setattr("file_agent.hf_batch.process_hf_qa_record", fail_if_called)
    resumed_run = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        resume=True,
    )

    assert resumed_run.durations_seconds == first_run.durations_seconds


def test_generate_hf_qa_records_treats_legacy_checkpoints_without_duration_as_unknown(
    monkeypatch, tmp_path
):
    install_fake_processor(monkeypatch, [])
    generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    # Simulate a checkpoint written before duration_seconds existed.
    checkpoint_path = tmp_path / "checkpoints" / "000000.json"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    del payload["duration_seconds"]
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("Processor must not be called for completed rows")

    monkeypatch.setattr("file_agent.hf_batch.process_hf_qa_record", fail_if_called)
    resumed_run = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        resume=True,
    )

    assert resumed_run.durations_seconds[0] is None
    assert resumed_run.durations_seconds[1] is not None


def test_duration_stats_summarizes_known_durations_and_ignores_unknown():
    stats = duration_stats([1.0, None, 3.0, 2.0])

    assert stats == {
        "count": 3,
        "total_seconds": 6.0,
        "average_seconds": 2.0,
        "min_seconds": 1.0,
        "max_seconds": 3.0,
    }


def test_duration_stats_handles_no_known_durations():
    stats = duration_stats([None, None])

    assert stats == {
        "count": 0,
        "total_seconds": None,
        "average_seconds": None,
        "min_seconds": None,
        "max_seconds": None,
    }


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


def test_cached_document_loader_dedupes_identical_content_at_different_paths(
    monkeypatch,
    tmp_path,
):
    """Regression test: some HF datasets store a separate per-question copy
    of the same source document (e.g. q0001/report.pdf, q0002/report.pdf -
    distinct Hub blobs, byte-identical content). A cache keyed by resolved
    path treats these as different files and re-parses every single row;
    the fix keys by content hash instead, so identical bytes at different
    paths still hit the cache."""
    first_path = tmp_path / "q0001" / "shared.txt"
    second_path = tmp_path / "q0002" / "shared.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("Shared document", encoding="utf-8")
    second_path.write_text("Shared document", encoding="utf-8")
    load_calls = []

    def fake_load_documents(file_paths):
        paths = list(file_paths)
        load_calls.append(paths)
        return [
            Document(
                file_name=paths[0].name,
                file_type="txt",
                blocks=[Block(id="block-1", text="Shared document", type="text", metadata={})],
            )
        ]

    monkeypatch.setattr("file_agent.hf_batch.load_documents", fake_load_documents)
    loader = _create_cached_document_loader()

    loader([first_path])
    loader([second_path])

    assert load_calls == [[first_path]]


def test_cached_document_loader_single_flights_concurrent_same_document_requests(
    monkeypatch, tmp_path
):
    """Two threads requesting the same document at the same time must trigger
    exactly one real parse - the second thread waits on the first thread's
    in-flight Future and gets its result, rather than calling load_documents
    a second time (see the cross-document non-blocking test below for the
    actual regression a global lock introduced)."""
    import threading as threading_module

    document_path = tmp_path / "shared.txt"
    document_path.write_text("Shared document", encoding="utf-8")
    load_calls = []
    parse_started = threading_module.Event()
    release_parse = threading_module.Event()

    def fake_load_documents(file_paths):
        paths = list(file_paths)
        load_calls.append(paths)
        parse_started.set()
        assert release_parse.wait(timeout=5), "second thread never let the parse proceed"
        return [
            Document(
                file_name=paths[0].name,
                file_type="txt",
                blocks=[Block(id="block-1", text="Shared document", type="text", metadata={})],
            )
        ]

    monkeypatch.setattr("file_agent.hf_batch.load_documents", fake_load_documents)
    loader = _create_cached_document_loader()

    results = []
    first_thread = threading_module.Thread(target=lambda: results.append(loader([document_path])))
    first_thread.start()
    assert parse_started.wait(timeout=5), "first thread never started parsing"

    # The second thread must not need a second parse to proceed - it should
    # be waiting on the first thread's in-flight Future, not calling
    # load_documents again (which would deadlock here, since fake_load_documents
    # only unblocks once per release_parse.set() below).
    second_thread = threading_module.Thread(target=lambda: results.append(loader([document_path])))
    second_thread.start()
    release_parse.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert len(load_calls) == 1
    assert len(results) == 2


def test_cached_document_loader_does_not_block_unrelated_documents_on_each_other(
    monkeypatch, tmp_path
):
    """Regression test: runs/local-qwen35-27b-agent-007-e5large-vlm-pilot's
    pilot showed 3 concurrent rows sharing one document all landing at
    ~410s versus ~125-150s for rows with unique documents - traced to a
    single global lock wrapped around the whole parse, which serialized
    every row against every other row regardless of which document they
    referenced, not just rows that actually shared one. A second thread
    parsing its own, unrelated document must complete without waiting for a
    first thread's still-in-progress, unrelated parse."""
    import threading as threading_module

    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("First document", encoding="utf-8")
    second_path.write_text("Second document", encoding="utf-8")

    first_parse_started = threading_module.Event()
    release_first_parse = threading_module.Event()

    def fake_load_documents(file_paths):
        paths = list(file_paths)
        if paths[0].name == "first.txt":
            first_parse_started.set()
            assert release_first_parse.wait(timeout=5), "first parse was never released"
        return [
            Document(
                file_name=paths[0].name,
                file_type="txt",
                blocks=[Block(id="block-1", text="doc", type="text", metadata={})],
            )
        ]

    monkeypatch.setattr("file_agent.hf_batch.load_documents", fake_load_documents)
    loader = _create_cached_document_loader()

    first_thread = threading_module.Thread(target=lambda: loader([first_path]))
    first_thread.start()
    assert first_parse_started.wait(timeout=5), "first thread never started parsing"

    second_results = []
    second_thread = threading_module.Thread(
        target=lambda: second_results.append(loader([second_path]))
    )
    second_thread.start()
    second_thread.join(timeout=5)

    # A global lock around the whole parse (the bug this fixes) would have
    # blocked this unrelated document behind the still-open first parse,
    # leaving the thread alive past the join timeout.
    assert not second_thread.is_alive(), "unrelated document's parse was blocked"
    assert len(second_results) == 1

    release_first_parse.set()
    first_thread.join(timeout=5)


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


def test_generate_hf_qa_records_continues_past_a_row_failure(monkeypatch, tmp_path):
    calls = []
    install_fake_processor(monkeypatch, calls, fail_on_id="q0002")

    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    # The batch completes in a single call - q0002 failing doesn't abort it.
    assert {record_id for record_id, _ in calls} == {"q0001", "q0002"}
    assert result.processed_count == 1
    assert result.failed_count == 1
    assert result.resumed_count == 0
    assert [record.id for record in result.records] == ["q0001", "q0002"]

    failed_record = result.records[1]
    assert failed_record.answer_model.startswith("[GENERATION FAILED:")
    assert "Failed on q0002" in failed_record.answer_model
    assert failed_record.contexts == ()

    # Only the successful row is checkpointed - the failed one stays retryable.
    assert (tmp_path / "checkpoints" / "000000.json").exists()
    assert not (tmp_path / "checkpoints" / "000001.json").exists()


def test_generate_hf_qa_records_retries_a_failed_row_on_resume(monkeypatch, tmp_path):
    install_fake_processor(monkeypatch, [], fail_on_id="q0002")
    generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
    )

    resumed_calls = []
    install_fake_processor(monkeypatch, resumed_calls)  # succeeds for every row now
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
    assert result.failed_count == 0
    assert [record.id for record in result.records] == ["q0001", "q0002"]
    assert result.records[1].answer_model == "q0002 generated"


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


def test_generate_hf_qa_records_output_order_survives_out_of_order_completion(
    monkeypatch, tmp_path
):
    """Rows now process concurrently (max_concurrency) and can finish in any
    order - q0001 is made to finish after q0002 here, on purpose, to prove
    the final result is still reassembled by row_index rather than by
    whichever order process_hf_qa_record happened to return."""
    completion_order = []

    def fake_process_hf_qa_record(**kwargs):
        record = kwargs["record"]
        if record.id == "q0001":
            time.sleep(0.05)
        completion_order.append(record.id)
        return make_generated_record(record)

    monkeypatch.setattr(
        "file_agent.hf_batch.process_hf_qa_record",
        fake_process_hf_qa_record,
    )

    result = generate_hf_qa_records(
        dataset=make_dataset(),
        dataset_id="owner/rag-qa",
        llm_client=DummyLLM(),
        output_dir=tmp_path,
        max_concurrency=2,
    )

    assert completion_order == ["q0002", "q0001"]
    assert [record.id for record in result.records] == ["q0001", "q0002"]


def test_generate_hf_qa_records_rejects_non_positive_max_concurrency(monkeypatch, tmp_path):
    install_fake_processor(monkeypatch, [])

    with pytest.raises(ValueError, match="max_concurrency"):
        generate_hf_qa_records(
            dataset=make_dataset(),
            dataset_id="owner/rag-qa",
            llm_client=DummyLLM(),
            output_dir=tmp_path,
            max_concurrency=0,
        )
