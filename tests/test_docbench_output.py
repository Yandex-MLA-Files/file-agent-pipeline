import json
from pathlib import Path

from datasets import Dataset, load_from_disk

from file_agent.docbench_batch import DocBenchFailure, DocBenchGeneratedRecord
from file_agent.docbench_output import save_docbench_artifacts
from file_agent.hf_rag import RetrievedContext


def _record() -> DocBenchGeneratedRecord:
    return DocBenchGeneratedRecord(
        id="docbench-000-q0001",
        folder_id=0,
        question_index=0,
        question="Question?",
        reference_answer="Gold",
        question_type="text-only",
        evidence="Evidence",
        domain="academia",
        source_file="doc.pdf",
        answer_model="Generated",
        contexts=(
            RetrievedContext(
                rank=1,
                chunk_id="c1",
                document_id="0/doc.pdf",
                text="Context",
                retrieval_text="Context",
                score=0.9,
                metadata_json='{"source_file": "doc.pdf"}',
            ),
        ),
        rag_mode="tool_agent",
        stop_reason="tool_agent_completed",
        search_queries=("Question",),
        tool_calls_json='[{"name": "search_documents"}]',
        llm_calls=2,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        vlm_calls=1,
        vlm_prompt_tokens=10,
        vlm_completion_tokens=5,
        vlm_total_tokens=15,
        document_vlm_calls=2,
        document_vlm_prompt_tokens=20,
        document_vlm_completion_tokens=10,
        document_vlm_total_tokens=30,
        answer_duration_seconds=2.0,
        document_preparation_seconds=4.0,
        amortized_duration_seconds=3.0,
        document_blocks=10,
        document_chunks=5,
        document_pages=2,
    )


def test_complete_output_contains_judge_ready_jsonl_and_reloadable_datasets(
    tmp_path: Path,
) -> None:
    artifacts = save_docbench_artifacts(
        records=[_record()],
        failures=[],
        output_dir=tmp_path,
        expected_count=1,
    )

    assert artifacts.is_complete
    assert artifacts.parquet_path.name == "answers.parquet"
    assert (
        Dataset.from_parquet(
            str(artifacts.parquet_path),
            cache_dir=str(tmp_path / "datasets-cache"),
        ).num_rows
        == 1
    )
    assert load_from_disk(artifacts.hf_dataset_path).num_rows == 1
    evaluation = json.loads(artifacts.evaluation_input_path.read_text(encoding="utf-8"))
    assert evaluation["sys_ans"] == "Generated"
    assert evaluation["answer"] == "Gold"
    assert evaluation["evidence"] == "Evidence"
    assert evaluation["type"] == "text-only"
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["usage"]["llm_calls"] == 2
    assert summary["usage"]["tool_calls"] == 1


def test_incomplete_output_is_marked_partial_and_lists_failure(tmp_path: Path) -> None:
    failure_path = tmp_path / "failures" / "failed.json"
    failure_path.parent.mkdir()
    failure_path.write_text("{}", encoding="utf-8")
    failure = DocBenchFailure(
        id="docbench-000-q0002",
        folder_id=0,
        error_type="TimeoutError",
        message="timeout",
        failure_path=failure_path,
    )

    artifacts = save_docbench_artifacts(
        records=[_record()],
        failures=[failure],
        output_dir=tmp_path,
        expected_count=2,
    )

    assert not artifacts.is_complete
    assert artifacts.parquet_path.name == "partial_answers.parquet"
    assert not (tmp_path / "answers.parquet").exists()
    incomplete = json.loads(artifacts.incomplete_rows_path.read_text(encoding="utf-8"))
    assert incomplete["pending_ids"] == ["docbench-000-q0002"]
