import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from file_agent.hf_batch import (
    BatchGenerationResult,
    build_generation_parameters,
    generate_hf_qa_records,
)
from file_agent.hf_dataset import QADatasetRecord, load_qa_dataset
from file_agent.hf_output import GeneratedDatasetArtifacts, save_generated_qa_dataset
from file_agent.llm.base import LLMClient
from file_agent.llm.factory import create_llm_client

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILE_NAME = "run_manifest.json"
FINAL_ARTIFACT_NAMES = ("answers.parquet", "hf_dataset", MANIFEST_FILE_NAME)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HFGenerationConfig:
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
    use_router: bool = False

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
class HFGenerationRunResult:
    batch: BatchGenerationResult
    artifacts: GeneratedDatasetArtifacts
    manifest_path: Path


def run_hf_dataset_generation(
    config: HFGenerationConfig,
    llm_client: LLMClient | None = None,
) -> HFGenerationRunResult:
    """Run dataset loading, RAG generation, final export, and manifest writing."""
    _validate_output_directory(config.output_dir)

    active_llm_client = (
        llm_client if llm_client is not None else create_llm_client(env_file=config.env_file)
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
        "Starting generation for %s rows from %s",
        len(selected_dataset),
        config.dataset_id,
    )
    batch_result = generate_hf_qa_records(
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
        use_router=config.use_router,
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

    LOGGER.info("Generation complete: %s rows", batch_result.total_count)
    return HFGenerationRunResult(
        batch=batch_result,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RAG answers for a Hugging Face QA dataset.",
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
        "--use-router",
        action="store_true",
        help="Route through the query classifier + planner (v1) instead of plain RAG (v0)",
    )
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

    try:
        config = HFGenerationConfig(
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
            use_router=args.use_router,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_hf_dataset_generation(config)
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
    config: HFGenerationConfig,
    dataset: Dataset,
    available_rows: int,
    llm_client: LLMClient,
    batch_result: BatchGenerationResult,
    artifacts: GeneratedDatasetArtifacts,
) -> dict[str, Any]:
    generation_parameters = build_generation_parameters(
        dataset_id=config.dataset_id,
        revision=config.revision,
        llm_client=llm_client,
        retriever=None,
        top_k=config.top_k,
        max_chars=config.max_chars,
        overlap=config.overlap,
        use_router=config.use_router,
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
