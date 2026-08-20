import json
import shutil
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import Dataset

from file_agent.docbench_batch import DocBenchFailure, DocBenchGeneratedRecord


@dataclass(frozen=True)
class DocBenchArtifacts:
    parquet_path: Path
    hf_dataset_path: Path
    predictions_path: Path
    evaluation_input_path: Path
    summary_path: Path
    incomplete_rows_path: Path | None
    row_count: int
    is_complete: bool


def save_docbench_artifacts(
    *,
    records: Sequence[DocBenchGeneratedRecord],
    failures: Sequence[DocBenchFailure],
    output_dir: str | Path,
    expected_count: int,
) -> DocBenchArtifacts:
    """Publish final artifacts only for a complete run; partials stay clearly marked."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    is_complete = not failures and len(records) == expected_count
    prefix = "" if is_complete else "partial_"
    parquet_path = output_path / f"{prefix}answers.parquet"
    hf_dataset_path = output_path / f"{prefix}hf_dataset"
    predictions_path = output_path / f"{prefix}predictions.jsonl"
    evaluation_input_path = output_path / f"{prefix}docbench_eval_input.jsonl"
    summary_path = output_path / f"{prefix}summary.json"

    rows = [record.to_dict() for record in records]
    dataset = Dataset.from_list(rows) if rows else Dataset.from_dict({"id": []})
    _write_parquet_atomic(dataset, parquet_path)
    _write_hf_dataset_atomic(dataset, hf_dataset_path)
    _write_jsonl_atomic(predictions_path, rows)
    _write_jsonl_atomic(
        evaluation_input_path,
        [_evaluation_row(record) for record in records],
    )
    _write_json_atomic(
        summary_path,
        build_docbench_summary(
            records=records,
            failures=failures,
            expected_count=expected_count,
        ),
    )

    incomplete_rows_path: Path | None = None
    if is_complete:
        stale_incomplete = output_path / "incomplete_rows.json"
        if stale_incomplete.exists():
            stale_incomplete.unlink()
        _remove_partial_artifacts(output_path)
    else:
        incomplete_rows_path = output_path / "incomplete_rows.json"
        _write_json_atomic(
            incomplete_rows_path,
            {
                "expected_count": expected_count,
                "completed_count": len(records),
                "failed_count": len(failures),
                "pending_ids": [failure.id for failure in failures],
                "failures": [
                    {
                        "id": failure.id,
                        "folder_id": failure.folder_id,
                        "error_type": failure.error_type,
                        "message": failure.message,
                        "details": str(failure.failure_path),
                    }
                    for failure in failures
                ],
            },
        )

    return DocBenchArtifacts(
        parquet_path=parquet_path,
        hf_dataset_path=hf_dataset_path,
        predictions_path=predictions_path,
        evaluation_input_path=evaluation_input_path,
        summary_path=summary_path,
        incomplete_rows_path=incomplete_rows_path,
        row_count=len(records),
        is_complete=is_complete,
    )


def build_docbench_summary(
    *,
    records: Sequence[DocBenchGeneratedRecord],
    failures: Sequence[DocBenchFailure],
    expected_count: int,
) -> dict[str, Any]:
    answer_latencies = [record.answer_duration_seconds for record in records]
    amortized_latencies = [record.amortized_duration_seconds for record in records]
    document_preparation_by_folder: dict[int, float] = {}
    document_vlm_by_folder: dict[int, tuple[int, int, int, int]] = {}
    for record in records:
        document_preparation_by_folder[record.folder_id] = max(
            document_preparation_by_folder.get(record.folder_id, 0.0),
            record.document_preparation_seconds,
        )
        document_vlm_by_folder[record.folder_id] = max(
            document_vlm_by_folder.get(record.folder_id, (0, 0, 0, 0)),
            (
                record.document_vlm_calls,
                record.document_vlm_prompt_tokens,
                record.document_vlm_completion_tokens,
                record.document_vlm_total_tokens,
            ),
        )

    ingestion_vlm_calls = sum(value[0] for value in document_vlm_by_folder.values())
    ingestion_vlm_prompt = sum(value[1] for value in document_vlm_by_folder.values())
    ingestion_vlm_completion = sum(value[2] for value in document_vlm_by_folder.values())
    ingestion_vlm_total = sum(value[3] for value in document_vlm_by_folder.values())
    answer_vlm_calls = sum(record.vlm_calls for record in records)
    answer_vlm_prompt = sum(record.vlm_prompt_tokens for record in records)
    answer_vlm_completion = sum(record.vlm_completion_tokens for record in records)
    answer_vlm_total = sum(record.vlm_total_tokens for record in records)

    return {
        "expected_questions": expected_count,
        "completed_questions": len(records),
        "failed_questions": len(failures),
        "is_complete": not failures and len(records) == expected_count,
        "documents": len({record.folder_id for record in records}),
        "by_domain": dict(sorted(Counter(record.domain for record in records).items())),
        "by_question_type": dict(
            sorted(Counter(record.question_type for record in records).items())
        ),
        "stop_reasons": dict(sorted(Counter(record.stop_reason for record in records).items())),
        "latency_seconds": {
            "answer_only": _distribution(answer_latencies),
            "amortized_with_document_preparation": _distribution(amortized_latencies),
            "document_preparation_total": sum(document_preparation_by_folder.values()),
        },
        "usage": {
            "llm_calls": sum(record.llm_calls for record in records),
            "prompt_tokens": sum(record.prompt_tokens for record in records),
            "completion_tokens": sum(record.completion_tokens for record in records),
            "total_tokens": sum(record.total_tokens for record in records),
            "vlm_calls": answer_vlm_calls + ingestion_vlm_calls,
            "vlm_prompt_tokens": answer_vlm_prompt + ingestion_vlm_prompt,
            "vlm_completion_tokens": answer_vlm_completion + ingestion_vlm_completion,
            "vlm_total_tokens": answer_vlm_total + ingestion_vlm_total,
            "vlm_answer_calls": answer_vlm_calls,
            "vlm_ingestion_calls": ingestion_vlm_calls,
            "tool_calls": sum(_tool_call_count(record.tool_calls_json) for record in records),
        },
        "failed_ids": [failure.id for failure in failures],
    }


def _evaluation_row(record: DocBenchGeneratedRecord) -> dict[str, Any]:
    # These names match the fields consumed by DocBench's evaluator, while the
    # stable id/file fields make joins deterministic and remove the fragile
    # numbered-text parsing used by the original demo runner.
    return {
        "id": record.id,
        "file": str(record.folder_id),
        "question_index": record.question_index,
        "question": record.question,
        "sys_ans": record.answer_model,
        "answer": record.reference_answer,
        "evidence": record.evidence,
        "type": record.question_type,
        "domain": record.domain,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _tool_call_count(value: str) -> int:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def _write_parquet_atomic(dataset: Dataset, path: Path) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        dataset.to_parquet(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_hf_dataset_atomic(dataset: Dataset, path: Path) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    backup = path.parent / f".{path.name}.{uuid4().hex}.bak"
    try:
        dataset.save_to_disk(temporary)
        if path.exists():
            path.replace(backup)
        temporary.replace(path)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not path.exists() and backup.exists():
            backup.replace(path)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            shutil.rmtree(backup)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_partial_artifacts(output_path: Path) -> None:
    for name in (
        "partial_answers.parquet",
        "partial_hf_dataset",
        "partial_predictions.jsonl",
        "partial_docbench_eval_input.jsonl",
        "partial_summary.json",
    ):
        path = output_path / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
