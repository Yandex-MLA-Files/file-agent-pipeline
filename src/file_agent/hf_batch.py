import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

from file_agent.agent.loop import MAX_ITERATIONS_DEFAULT
from file_agent.agent.tools import ALL_TOOL_NAMES
from file_agent.document import Document
from file_agent.hf_dataset import QADatasetRecord, validate_qa_dataset
from file_agent.hf_rag import DocumentLoader, GeneratedQARecord, process_hf_qa_record
from file_agent.lancedb_retriever import DEFAULT_SEMANTIC_MODEL_NAME
from file_agent.llm.base import LLMClient
from file_agent.qa import build_qa_prompt
from file_agent.rag import load_documents
from file_agent.retrieval import Retriever

CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
RAG_PIPELINE_VERSION = "section-token-small-to-big-v1"
AGENT_PIPELINE_VERSION = "react-tool-calling-v2"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchGenerationResult:
    records: tuple[GeneratedQARecord, ...]
    processed_count: int
    resumed_count: int
    # Rows where process_hf_qa_record raised (a genuine LLM/pipeline
    # failure, not a tool-call error - those are already handled inside the
    # agent loop). Included in records as a placeholder (see _failed_record)
    # so the row count still matches the source dataset, but never
    # checkpointed - a later --resume retries it fresh rather than
    # permanently baking in what might have been a transient failure.
    failed_count: int = 0
    # One entry per record, same order - None where a row's timing is
    # unknown (a checkpoint written before this field existed, on --resume).
    durations_seconds: tuple[float | None, ...] = field(default_factory=tuple)

    @property
    def total_count(self) -> int:
        return len(self.records)


def duration_stats(durations: Sequence[float | None]) -> dict[str, Any]:
    """Aggregate per-row timings, ignoring rows with unknown duration."""
    known = [duration for duration in durations if duration is not None]
    if not known:
        return {
            "count": 0,
            "total_seconds": None,
            "average_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(known),
        "total_seconds": round(sum(known), 2),
        "average_seconds": round(sum(known) / len(known), 2),
        "min_seconds": round(min(known), 2),
        "max_seconds": round(max(known), 2),
    }


def generate_hf_qa_records(
    dataset: Dataset,
    dataset_id: str,
    llm_client: LLMClient,
    output_dir: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    resume: bool = False,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    verify_answers: bool = True,
) -> BatchGenerationResult:
    validate_qa_dataset(dataset)
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")

    checkpoints_dir = Path(output_dir) / CHECKPOINTS_DIRECTORY_NAME
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_checkpoint_files(
        checkpoints_dir=checkpoints_dir,
        dataset_size=len(dataset),
        resume=resume,
    )

    parameters = build_generation_parameters(
        dataset_id=dataset_id,
        revision=revision,
        llm_client=llm_client,
        retriever=retriever,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        max_iterations=max_iterations,
        verify_answers=verify_answers,
    )
    records: list[GeneratedQARecord] = []
    durations: list[float | None] = []
    processed_count = 0
    resumed_count = 0
    failed_count = 0
    document_loader = _create_cached_document_loader()

    for row_index, row in enumerate(dataset):
        record = QADatasetRecord.from_row(row, row_index=row_index)
        checkpoint_path = _checkpoint_path(checkpoints_dir, row_index)

        if checkpoint_path.exists():
            generated_record, duration_seconds = _load_checkpoint(
                checkpoint_path=checkpoint_path,
                source_record=record,
                expected_parameters=parameters,
            )
            resumed_count += 1
            LOGGER.info(
                "Loaded checkpoint for row %s/%s (%s)",
                row_index + 1,
                len(dataset),
                record.id,
            )
        else:
            started_at = time.monotonic()
            try:
                generated_record = process_hf_qa_record(
                    record=record,
                    dataset_id=dataset_id,
                    llm_client=llm_client,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    top_k=top_k,
                    max_chars=max_chars,
                    overlap=overlap,
                    retriever=retriever,
                    document_loader=document_loader,
                    max_iterations=max_iterations,
                    verify_answers=verify_answers,
                )
            except Exception as exc:  # noqa: BLE001 - row-level isolation is deliberate
                # A genuine LLM/pipeline failure on one row (e.g. the model
                # exhausting its empty-response retries) must not abort the
                # whole batch - it becomes a visible placeholder instead, and
                # is left uncheckpointed so --resume retries it fresh (it
                # might have been transient; temperature > 0 means a retry
                # can actually diverge from the same failure).
                duration_seconds = time.monotonic() - started_at
                LOGGER.error(
                    "Row %s/%s (%s) failed after %.2fs: %s",
                    row_index + 1,
                    len(dataset),
                    record.id,
                    duration_seconds,
                    exc,
                    exc_info=True,
                )
                generated_record = _failed_record(record, exc)
                failed_count += 1
            else:
                duration_seconds = time.monotonic() - started_at
                _validate_generated_record(generated_record, record)
                _write_checkpoint(
                    checkpoint_path=checkpoint_path,
                    parameters=parameters,
                    generated_record=generated_record,
                    duration_seconds=duration_seconds,
                )
                processed_count += 1
                LOGGER.info(
                    "Processed row %s/%s (%s) in %.2fs",
                    row_index + 1,
                    len(dataset),
                    record.id,
                    duration_seconds,
                )

        records.append(generated_record)
        durations.append(duration_seconds)

    return BatchGenerationResult(
        records=tuple(records),
        processed_count=processed_count,
        resumed_count=resumed_count,
        failed_count=failed_count,
        durations_seconds=tuple(durations),
    )


def _failed_record(record: QADatasetRecord, exc: Exception) -> GeneratedQARecord:
    return GeneratedQARecord(
        id=record.id,
        question=record.question,
        doc_ids=record.doc_ids,
        answer_model=f"[GENERATION FAILED: {exc}]",
        contexts=(),
        answer=record.answer,
    )


def _create_cached_document_loader() -> DocumentLoader:
    cache: dict[str, Document] = {}

    def load_cached_documents(file_paths: list[str | Path]) -> list[Document]:
        documents: list[Document] = []

        for file_path in file_paths:
            source_path = Path(file_path)
            # Content hash, not resolved path: some HF datasets store a
            # separate per-question copy of the same source document (e.g.
            # q0001/report.pdf, q0002/report.pdf - distinct Hub blobs, byte-
            # identical content), which gave every row its own path and made
            # a path-keyed cache miss every single time.
            cache_key = hashlib.sha256(source_path.read_bytes()).hexdigest()
            cached_document = cache.get(cache_key)
            if cached_document is None:
                cached_document = load_documents([source_path])[0]
                cache[cache_key] = cached_document
            else:
                LOGGER.info("Reusing parsed document from batch cache: %s", source_path.name)

            # HF processing adds row-specific dataset metadata to every block.
            # Return an isolated copy so one question cannot mutate the cached
            # document or leak its metadata into another question.
            documents.append(deepcopy(cached_document))

        return documents

    return load_cached_documents


def build_generation_parameters(
    dataset_id: str,
    revision: str | None,
    llm_client: LLMClient,
    retriever: Retriever | None,
    top_k: int,
    max_chars: int,
    overlap: int,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    verify_answers: bool = True,
) -> dict[str, Any]:
    prompt_template = build_qa_prompt(
        question="{question}",
        context="{context}",
    )
    return {
        "dataset_id": dataset_id,
        "revision": revision,
        "model_id": _model_identifier(llm_client),
        "temperature": _optional_scalar_attribute(llm_client, "temperature"),
        "max_tokens": _optional_scalar_attribute(llm_client, "max_tokens"),
        "retriever": _component_identifier(retriever) if retriever is not None else "default",
        "rag_pipeline_version": RAG_PIPELINE_VERSION,
        "agent_pipeline_version": AGENT_PIPELINE_VERSION,
        "max_iterations": max_iterations,
        "verify_answers": verify_answers,
        "tool_names": list(ALL_TOOL_NAMES),
        "embedding_model": os.getenv("EMBEDDING_MODEL") or DEFAULT_SEMANTIC_MODEL_NAME,
        "ocr_engine": os.getenv("OCR_ENGINE", "easyocr").strip().lower(),
        "ocr_langs": os.getenv("OCR_LANGS", "ru,en").strip(),
        "vlm_backend": os.getenv("VLM_BACKEND", "off").strip().lower(),
        "vlm_model": _vlm_model_identifier(),
        "top_k": top_k,
        "max_chars": max_chars,
        "overlap": overlap,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
    }


def _vlm_model_identifier() -> str | None:
    backend = os.getenv("VLM_BACKEND", "off").strip().lower()
    if backend == "smolvlm":
        return os.getenv(
            "VLM_LOCAL_MODEL",
            "HuggingFaceTB/SmolVLM-256M-Instruct",
        )
    if backend == "openai":
        return os.getenv("VLM_MODEL")
    return None


def _model_identifier(llm_client: LLMClient) -> str:
    model = getattr(llm_client, "model", None)
    if isinstance(model, str) and model.strip():
        return model
    return _component_identifier(llm_client)


def _component_identifier(component: object) -> str:
    component_type = type(component)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _optional_scalar_attribute(component: object, name: str) -> str | int | float | bool | None:
    value = getattr(component, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_existing_checkpoint_files(
    checkpoints_dir: Path,
    dataset_size: int,
    resume: bool,
) -> None:
    existing_paths = sorted(checkpoints_dir.glob("*.json"))
    if existing_paths and not resume:
        raise FileExistsError(
            f"Checkpoints already exist in {checkpoints_dir}; use resume=True or a new output_dir"
        )

    expected_names = {_checkpoint_filename(row_index) for row_index in range(dataset_size)}
    unexpected_names = [path.name for path in existing_paths if path.name not in expected_names]
    if unexpected_names:
        raise ValueError(
            "Checkpoint directory contains rows outside the current dataset: "
            + ", ".join(unexpected_names)
        )


def _checkpoint_filename(row_index: int) -> str:
    return f"{row_index:06d}.json"


def _checkpoint_path(checkpoints_dir: Path, row_index: int) -> Path:
    return checkpoints_dir / _checkpoint_filename(row_index)


def _write_checkpoint(
    checkpoint_path: Path,
    parameters: dict[str, Any],
    generated_record: GeneratedQARecord,
    duration_seconds: float,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "parameters": parameters,
        "result": generated_record.to_dict(),
        # Sibling to "result", not part of it or CHECKPOINT_SCHEMA_VERSION -
        # purely-additive run metadata, not part of the answer's own shape.
        "duration_seconds": round(duration_seconds, 3),
    }
    temporary_path = checkpoint_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    source_record: QADatasetRecord,
    expected_parameters: dict[str, Any],
) -> tuple[GeneratedQARecord, float | None]:
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read checkpoint: {checkpoint_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a JSON object: {checkpoint_path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {checkpoint_path}")
    if payload.get("parameters") != expected_parameters:
        raise ValueError(f"Checkpoint parameters do not match the current run: {checkpoint_path}")

    result_value = payload.get("result")
    if not isinstance(result_value, dict):
        raise ValueError(f"Checkpoint result must be a JSON object: {checkpoint_path}")

    generated_record = GeneratedQARecord.from_dict(result_value)
    _validate_generated_record(generated_record, source_record)

    # None for checkpoints written before this field existed (or otherwise
    # malformed) - unknown timing, not zero, so it's excluded from stats
    # rather than skewing the average down.
    duration_value = payload.get("duration_seconds")
    is_number = isinstance(duration_value, (int, float)) and not isinstance(duration_value, bool)
    duration_seconds = duration_value if is_number else None
    return generated_record, duration_seconds


def _validate_generated_record(
    generated_record: GeneratedQARecord,
    source_record: QADatasetRecord,
) -> None:
    if generated_record.id != source_record.id:
        raise ValueError(f"Generated id does not match source record: {source_record.id}")
    if generated_record.question != source_record.question:
        raise ValueError(f"Generated question does not match source record: {source_record.id}")
    if generated_record.doc_ids != source_record.doc_ids:
        raise ValueError(f"Generated doc_ids do not match source record: {source_record.id}")
    if generated_record.answer != source_record.answer:
        raise ValueError(f"Generated gold answer does not match source record: {source_record.id}")
