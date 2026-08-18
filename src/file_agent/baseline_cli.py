import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from file_agent.hf_batch import (
    _component_identifier,
    _create_cached_document_loader,
    _model_identifier,
    _optional_scalar_attribute,
    _vlm_model_identifier,
)
from file_agent.hf_dataset import (
    QADatasetRecord,
    download_record_documents,
    load_qa_dataset,
    validate_qa_dataset,
)
from file_agent.hf_output import GeneratedDatasetArtifacts, save_generated_qa_dataset
from file_agent.hf_rag import DocumentLoader, GeneratedQARecord, serialize_search_results
from file_agent.lancedb_retriever import DEFAULT_SEMANTIC_MODEL_NAME, LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.llm.factory import create_generation_llm_client
from file_agent.qa import build_qa_prompt
from file_agent.rag import answer_documents, load_documents
from file_agent.retrieval import Retriever
from file_agent.telemetry import configure_telemetry

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILE_NAME = "run_manifest.json"
FINAL_ARTIFACT_NAMES = ("answers.parquet", "hf_dataset", MANIFEST_FILE_NAME)
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
RAG_PIPELINE_VERSION = "section-token-small-to-big-v1"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchGenerationResult:
    records: tuple[GeneratedQARecord, ...]
    processed_count: int
    resumed_count: int

    @property
    def total_count(self) -> int:
        return len(self.records)


def process_baseline_hf_qa_record(
    record: QADatasetRecord,
    dataset_id: str,
    llm_client: LLMClient,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    document_loader: DocumentLoader | None = None,
) -> GeneratedQARecord:
    document_paths = download_record_documents(
        record=record,
        dataset_id=dataset_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return process_baseline_qa_record(
        record=record,
        document_paths=document_paths,
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        retriever=retriever,
        document_loader=document_loader,
    )


def process_baseline_qa_record(
    record: QADatasetRecord,
    document_paths: list[str | Path],
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    document_loader: DocumentLoader | None = None,
) -> GeneratedQARecord:
    if len(document_paths) != len(record.doc_ids):
        raise ValueError("document_paths count must match record.doc_ids count")

    active_document_loader = document_loader or load_documents
    documents = active_document_loader(document_paths)
    for document, doc_id in zip(documents, record.doc_ids, strict=True):
        for block in document.blocks:
            block.metadata["dataset_record_id"] = record.id
            block.metadata["dataset_doc_id"] = doc_id

    active_retriever = retriever if retriever is not None else LanceDBRetriever()
    try:
        response = answer_documents(
            documents=documents,
            question=record.question,
            llm_client=llm_client,
            top_k=top_k,
            max_chars=max_chars,
            overlap=overlap,
            retriever=active_retriever,
        )
        contexts = serialize_search_results(response.sources)

        return GeneratedQARecord(
            id=record.id,
            question=record.question,
            doc_ids=record.doc_ids,
            answer_model=response.answer,
            contexts=contexts,
            answer=record.answer,
        )
    finally:
        active_retriever.clear()


def generate_baseline_qa_records(
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

    parameters = build_baseline_parameters(
        dataset_id=dataset_id,
        revision=revision,
        llm_client=llm_client,
        retriever=retriever,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
    )
    records: list[GeneratedQARecord] = []
    processed_count = 0
    resumed_count = 0
    document_loader = _create_cached_document_loader()

    for row_index, row in enumerate(dataset):
        record = QADatasetRecord.from_row(row, row_index=row_index)
        checkpoint_path = _checkpoint_path(checkpoints_dir, row_index)

        if checkpoint_path.exists():
            generated_record = _load_checkpoint(
                checkpoint_path=checkpoint_path,
                source_record=record,
                expected_parameters=parameters,
            )
            resumed_count += 1
            LOGGER.info(
                "Loaded checkpoint for row %s/%s (%s)", row_index + 1, len(dataset), record.id
            )
        else:
            generated_record = process_baseline_hf_qa_record(
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
            )
            _validate_generated_record(generated_record, record)
            _write_checkpoint(
                checkpoint_path=checkpoint_path,
                parameters=parameters,
                generated_record=generated_record,
            )
            processed_count += 1
            LOGGER.info("Processed row %s/%s (%s)", row_index + 1, len(dataset), record.id)

        records.append(generated_record)

    return BatchGenerationResult(
        records=tuple(records),
        processed_count=processed_count,
        resumed_count=resumed_count,
    )


def build_baseline_parameters(
    dataset_id: str,
    revision: str | None,
    llm_client: LLMClient,
    retriever: Retriever | None,
    top_k: int,
    max_chars: int,
    overlap: int,
) -> dict[str, Any]:
    prompt_template = build_qa_prompt(question="{question}", context="{context}")
    return {
        "pipeline_kind": "plain_rag_baseline",
        "dataset_id": dataset_id,
        "revision": revision,
        "model_id": _model_identifier(llm_client),
        "temperature": _optional_scalar_attribute(llm_client, "temperature"),
        "max_tokens": _optional_scalar_attribute(llm_client, "max_tokens"),
        "retriever": _component_identifier(retriever) if retriever is not None else "default",
        "rag_pipeline_version": RAG_PIPELINE_VERSION,
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
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "parameters": parameters,
        "result": generated_record.to_dict(),
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
) -> GeneratedQARecord:
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
    return generated_record


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


@dataclass(frozen=True)
class BaselineGenerationConfig:
    dataset_id: str
    output_dir: Path
    config_name: str | None = None
    split: str = "train"
    revision: str | None = None
    cache_dir: Path | None = None
    env_file: Path = Path(".env")
    top_k: int = 5
    max_chars: int = 1000
    overlap: int = 100
    limit: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_id, "dataset_id")
        _require_non_empty(self.split, "split")
        if self.config_name is not None:
            _require_non_empty(self.config_name, "config_name")
        if self.revision is not None:
            _require_non_empty(self.revision, "revision")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        if self.overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        if self.overlap >= self.max_chars:
            raise ValueError("overlap must be smaller than max_chars")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0")


@dataclass(frozen=True)
class BaselineGenerationRunResult:
    batch: BatchGenerationResult
    artifacts: GeneratedDatasetArtifacts
    manifest_path: Path


def run_baseline_generation(
    config: BaselineGenerationConfig,
    llm_client: LLMClient | None = None,
) -> BaselineGenerationRunResult:
    """Run dataset loading, plain single-shot RAG generation, export, and manifest writing."""
    _validate_output_directory(config.output_dir)

    active_llm_client = (
        llm_client
        if llm_client is not None
        else create_generation_llm_client(env_file=config.env_file)
    )
    source_dataset = load_qa_dataset(
        dataset_id=config.dataset_id,
        config_name=config.config_name,
        split=config.split,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )
    available_rows = len(source_dataset)
    selected_dataset = _select_rows(source_dataset, config.limit)
    if not len(selected_dataset):
        raise ValueError("The selected dataset contains no rows")

    LOGGER.info(
        "Starting baseline generation for %s rows from %s",
        len(selected_dataset),
        config.dataset_id,
    )
    batch_result = generate_baseline_qa_records(
        dataset=selected_dataset,
        dataset_id=config.dataset_id,
        llm_client=active_llm_client,
        output_dir=config.output_dir,
        revision=config.revision,
        cache_dir=config.cache_dir,
        top_k=config.top_k,
        max_chars=config.max_chars,
        overlap=config.overlap,
        resume=config.resume,
    )
    artifacts = save_generated_qa_dataset(
        source_dataset=selected_dataset,
        records=batch_result.records,
        output_dir=config.output_dir,
    )
    manifest = _build_manifest(
        config=config,
        dataset=selected_dataset,
        available_rows=available_rows,
        llm_client=active_llm_client,
        batch_result=batch_result,
        artifacts=artifacts,
    )
    manifest_path = _write_manifest(config.output_dir, manifest)

    LOGGER.info("Baseline generation complete: %s rows", batch_result.total_count)
    return BaselineGenerationRunResult(
        batch=batch_result,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plain single-shot RAG answers (no agent, no tools) for a Hugging "
        "Face QA dataset - a baseline to compare against the ReAct agent pipeline.",
    )
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--config-name", help="Dataset configuration name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--revision", help="Dataset commit or revision")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="LLM environment file")
    parser.add_argument("--top-k", type=_positive_int, default=5, help="Retrieved chunks per row")
    parser.add_argument("--max-chars", type=_positive_int, default=1000, help="Chunk size")
    parser.add_argument("--overlap", type=_non_negative_int, default=100, help="Chunk overlap")
    parser.add_argument("--limit", type=_positive_int, help="Process only the first N rows")
    parser.add_argument("--resume", action="store_true", help="Reuse matching row checkpoints")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_telemetry()

    try:
        config = BaselineGenerationConfig(
            dataset_id=args.dataset_id,
            output_dir=args.output_dir,
            config_name=args.config_name,
            split=args.split,
            revision=args.revision,
            cache_dir=args.cache_dir,
            env_file=args.env_file,
            top_k=args.top_k,
            max_chars=args.max_chars,
            overlap=args.overlap,
            limit=args.limit,
            resume=args.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_baseline_generation(config)
    print(
        f"Completed {result.batch.total_count} rows "
        f"(processed: {result.batch.processed_count}, resumed: {result.batch.resumed_count})"
    )
    print(f"Parquet: {result.artifacts.parquet_path}")
    print(f"Hugging Face dataset: {result.artifacts.hf_dataset_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _select_rows(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _validate_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")

    existing_paths = [output_dir / name for name in FINAL_ARTIFACT_NAMES]
    existing_paths = [path for path in existing_paths if path.exists()]
    if existing_paths:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Final output artifacts already exist ({names}); use a new output directory"
        )


def _build_manifest(
    config: BaselineGenerationConfig,
    dataset: Dataset,
    available_rows: int,
    llm_client: LLMClient,
    batch_result: BatchGenerationResult,
    artifacts: GeneratedDatasetArtifacts,
) -> dict[str, Any]:
    generation_parameters = build_baseline_parameters(
        dataset_id=config.dataset_id,
        revision=config.revision,
        llm_client=llm_client,
        retriever=None,
        top_k=config.top_k,
        max_chars=config.max_chars,
        overlap=config.overlap,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_timestamp(),
        "dataset": {
            "id": config.dataset_id,
            "config_name": config.config_name,
            "split": config.split,
            "revision": config.revision,
            "available_rows": available_rows,
            "selected_rows": len(dataset),
            "limit": config.limit,
            "records_sha256": _dataset_records_sha256(dataset),
        },
        "generation": {
            **generation_parameters,
            "resume_requested": config.resume,
        },
        "result": {
            "total_count": batch_result.total_count,
            "processed_count": batch_result.processed_count,
            "resumed_count": batch_result.resumed_count,
        },
        "artifacts": {
            "parquet": artifacts.parquet_path.name,
            "hf_dataset": artifacts.hf_dataset_path.name,
        },
    }


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILE_NAME
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def _dataset_records_sha256(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for row_index, row in enumerate(dataset):
        record = QADatasetRecord.from_row(row, row_index=row_index)
        value = {
            "id": record.id,
            "question": record.question,
            "answer": record.answer,
            "doc_ids": list(record.doc_ids),
        }
        digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


if __name__ == "__main__":
    raise SystemExit(main())
