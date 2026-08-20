import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from file_agent.docbench_batch import (
    DocBenchBatchResult,
    build_docbench_generation_parameters,
    generate_docbench_records,
)
from file_agent.docbench_dataset import (
    DOCBENCH_DOMAINS,
    DOCBENCH_QUESTION_TYPES,
    DocBenchRecord,
    load_docbench_records,
    select_docbench_records,
)
from file_agent.docbench_output import DocBenchArtifacts, save_docbench_artifacts
from file_agent.llm.base import LLMClient
from file_agent.llm.factory import DEFAULT_MAX_TOKENS, create_llm_client
from file_agent.rag import resolve_max_tool_rounds, resolve_rag_mode
from file_agent.vlm.base import VLMClient
from file_agent.vlm.factory import create_vlm_client

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILE_NAME = "run_manifest.json"
RUN_CONFIG_FILE_NAME = "run_config.json"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocBenchGenerationConfig:
    data_dir: Path
    output_dir: Path
    env_file: Path = Path(".env")
    top_k: int = 5
    max_chars: int = 1000
    overlap: int = 100
    folder_start: int | None = None
    folder_end: int | None = None
    folder_ids: tuple[int, ...] = ()
    record_ids: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    question_types: tuple[str, ...] = ()
    limit: int | None = None
    resume: bool = False
    fail_fast: bool = False
    rag_mode: str | None = None
    max_tool_rounds: int | None = None
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        if self.overlap < 0 or self.overlap >= self.max_chars:
            raise ValueError("overlap must be non-negative and smaller than max_chars")
        for name, value in (
            ("folder_start", self.folder_start),
            ("folder_end", self.folder_end),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.folder_start is not None
            and self.folder_end is not None
            and self.folder_start > self.folder_end
        ):
            raise ValueError("folder_start must be less than or equal to folder_end")
        if any(folder_id < 0 for folder_id in self.folder_ids):
            raise ValueError("folder_ids must be non-negative")
        _require_unique(self.folder_ids, "folder_ids")
        _require_unique(self.record_ids, "record_ids")
        if any(not record_id.strip() for record_id in self.record_ids):
            raise ValueError("record_ids must contain only non-empty values")
        if any(domain not in DOCBENCH_DOMAINS for domain in self.domains):
            raise ValueError("domains contain an unsupported DocBench domain")
        if any(value not in DOCBENCH_QUESTION_TYPES for value in self.question_types):
            raise ValueError("question_types contain an unsupported DocBench type")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0")
        if self.rag_mode is not None:
            resolve_rag_mode(self.rag_mode)
        if self.max_tool_rounds is not None:
            resolve_max_tool_rounds(self.max_tool_rounds)
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")


@dataclass(frozen=True)
class DocBenchGenerationRunResult:
    batch: DocBenchBatchResult
    artifacts: DocBenchArtifacts
    manifest_path: Path
    selected_count: int
    available_count: int


def run_docbench_generation(
    config: DocBenchGenerationConfig,
    *,
    llm_client: LLMClient | None = None,
    vlm_client: VLMClient | None = None,
) -> DocBenchGenerationRunResult:
    """Load a local DocBench checkout, generate answers, and publish artifacts."""
    _validate_output_directory(config.output_dir, config.resume)
    # Parser modules may have loaded a project .env on import. This explicit
    # override makes --env-file authoritative for LLM, OCR, embeddings and VLM.
    load_dotenv(config.env_file, override=True)

    active_mode = resolve_rag_mode(config.rag_mode)
    active_max_tool_rounds = (
        resolve_max_tool_rounds(config.max_tool_rounds) if active_mode == "tool_agent" else None
    )
    active_llm = llm_client or create_llm_client(
        env_file=config.env_file,
        load_env=False,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    # The same VLM instance is reused for ingestion figure descriptions and
    # agent visual tools. This also makes its request/token counters meaningful.
    active_vlm = vlm_client if vlm_client is not None else create_vlm_client()

    all_records = load_docbench_records(config.data_dir)
    selected_records = select_docbench_records(
        all_records,
        folder_start=config.folder_start,
        folder_end=config.folder_end,
        folder_ids=config.folder_ids,
        record_ids=config.record_ids,
        domains=config.domains,
        question_types=config.question_types,
        limit=config.limit,
    )
    LOGGER.info(
        "Starting DocBench generation: %s selected questions in %s documents (%s available)",
        len(selected_records),
        len({record.folder_id for record in selected_records}),
        len(all_records),
    )
    parameters = build_docbench_generation_parameters(
        llm_client=active_llm,
        top_k=config.top_k,
        max_chars=config.max_chars,
        overlap=config.overlap,
        rag_mode=active_mode,
        max_tool_rounds=active_max_tool_rounds,
    )
    _validate_or_write_run_config(
        config=config,
        selected_records=selected_records,
        parameters=parameters,
    )
    batch = generate_docbench_records(
        records=selected_records,
        llm_client=active_llm,
        output_dir=config.output_dir,
        top_k=config.top_k,
        max_chars=config.max_chars,
        overlap=config.overlap,
        resume=config.resume,
        rag_mode=active_mode,
        max_tool_rounds=active_max_tool_rounds,
        vlm_client=active_vlm,
        fail_fast=config.fail_fast,
    )
    artifacts = save_docbench_artifacts(
        records=batch.records,
        failures=batch.failures,
        output_dir=config.output_dir,
        expected_count=len(selected_records),
    )
    manifest = _build_manifest(
        config=config,
        available_records=all_records,
        selected_records=selected_records,
        parameters=parameters,
        batch=batch,
        artifacts=artifacts,
    )
    manifest_path = _write_manifest(config.output_dir, manifest)
    return DocBenchGenerationRunResult(
        batch=batch,
        artifacts=artifacts,
        manifest_path=manifest_path,
        selected_count=len(selected_records),
        available_count=len(all_records),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable file-agent answers for a local DocBench checkout. "
            "A PDF is parsed and indexed once for all selected questions in its folder."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Path to DocBench/data (numeric document folders)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Environment file")
    parser.add_argument("--top-k", type=_positive_int, default=5, help="Retrieved chunks")
    parser.add_argument("--max-chars", type=_positive_int, default=1000, help="Chunk budget")
    parser.add_argument("--overlap", type=_non_negative_int, default=100, help="Chunk overlap")
    parser.add_argument("--folder-start", type=_non_negative_int, help="First folder, inclusive")
    parser.add_argument("--folder-end", type=_non_negative_int, help="Last folder, inclusive")
    parser.add_argument(
        "--folder-id",
        dest="folder_ids",
        action="append",
        type=_non_negative_int,
        default=[],
        help="Select one exact folder; repeat for multiple folders",
    )
    parser.add_argument(
        "--record-id",
        dest="record_ids",
        action="append",
        default=[],
        help="Select one stable record ID; repeat for multiple records",
    )
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=DOCBENCH_DOMAINS,
        default=[],
        help="Filter by official domain; repeat for multiple domains",
    )
    parser.add_argument(
        "--question-type",
        dest="question_types",
        action="append",
        choices=DOCBENCH_QUESTION_TYPES,
        default=[],
        help="Filter by raw DocBench question type; repeat for multiple types",
    )
    parser.add_argument("--limit", type=_positive_int, help="First N questions after filtering")
    parser.add_argument("--resume", action="store_true", help="Reuse validated row checkpoints")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed question instead of checkpointing and continuing",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("standard", "tool_agent"),
        help="RAG mode (default: RAG_MODE or standard)",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=_positive_int,
        help="Maximum tool-agent rounds (default: RAG_MAX_TOOL_ROUNDS or 4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum completion tokens per LLM request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console/file logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log path (default: <output-dir>/generation.log)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    log_file = args.log_file or args.output_dir / "generation.log"
    _configure_logging(args.log_level, log_file)

    try:
        config = DocBenchGenerationConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            env_file=args.env_file,
            top_k=args.top_k,
            max_chars=args.max_chars,
            overlap=args.overlap,
            folder_start=args.folder_start,
            folder_end=args.folder_end,
            folder_ids=tuple(args.folder_ids),
            record_ids=tuple(args.record_ids),
            domains=tuple(args.domains),
            question_types=tuple(args.question_types),
            limit=args.limit,
            resume=args.resume,
            fail_fast=args.fail_fast,
            rag_mode=args.rag_mode,
            max_tool_rounds=args.max_tool_rounds,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_docbench_generation(config)
    print(
        f"Completed {result.batch.total_count}/{result.selected_count} DocBench questions "
        f"(processed: {result.batch.processed_count}, resumed: {result.batch.resumed_count})"
    )
    print(f"Parquet: {result.artifacts.parquet_path}")
    print(f"Predictions: {result.artifacts.predictions_path}")
    print(f"DocBench evaluation input: {result.artifacts.evaluation_input_path}")
    print(f"Summary: {result.artifacts.summary_path}")
    print(f"Manifest: {result.manifest_path}")
    if not result.artifacts.is_complete:
        print(
            "Run is incomplete. Retry the same command with --resume. Details: "
            f"{result.artifacts.incomplete_rows_path}"
        )
        return 2
    return 0


def _build_manifest(
    *,
    config: DocBenchGenerationConfig,
    available_records: Sequence[DocBenchRecord],
    selected_records: Sequence[DocBenchRecord],
    parameters: Mapping[str, Any],
    batch: DocBenchBatchResult,
    artifacts: DocBenchArtifacts,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_timestamp(),
        "dataset": {
            "name": "Anni-Zou/DocBench",
            "data_dir": str(config.data_dir.resolve()),
            "available_questions": len(available_records),
            "available_documents": len({record.folder_id for record in available_records}),
            "selected_questions": len(selected_records),
            "selected_documents": len({record.folder_id for record in selected_records}),
            "selection_sha256": _selection_sha256(selected_records),
            "by_domain": dict(
                sorted(Counter(record.domain for record in selected_records).items())
            ),
            "by_question_type": dict(
                sorted(Counter(record.question_type for record in selected_records).items())
            ),
            "selection": {
                "folder_start": config.folder_start,
                "folder_end": config.folder_end,
                "folder_ids": list(config.folder_ids),
                "record_ids": list(config.record_ids),
                "domains": list(config.domains),
                "question_types": list(config.question_types),
                "limit": config.limit,
            },
        },
        "generation": {
            **parameters,
            "resume_requested": config.resume,
            "fail_fast": config.fail_fast,
        },
        "result": {
            "is_complete": artifacts.is_complete,
            "completed_count": batch.total_count,
            "processed_count": batch.processed_count,
            "resumed_count": batch.resumed_count,
            "failure_count": len(batch.failures),
            "failed_ids": [failure.id for failure in batch.failures],
        },
        "artifacts": {
            "parquet": artifacts.parquet_path.name,
            "hf_dataset": artifacts.hf_dataset_path.name,
            "predictions": artifacts.predictions_path.name,
            "docbench_eval_input": artifacts.evaluation_input_path.name,
            "summary": artifacts.summary_path.name,
            "run_config": RUN_CONFIG_FILE_NAME,
            "incomplete_rows": (
                artifacts.incomplete_rows_path.name if artifacts.incomplete_rows_path else None
            ),
        },
    }


def _validate_output_directory(output_dir: Path, resume: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")
    final_paths = [
        output_dir / name
        for name in (
            "answers.parquet",
            "hf_dataset",
            "predictions.jsonl",
            "docbench_eval_input.jsonl",
        )
        if (output_dir / name).exists()
    ]
    if final_paths and not resume:
        names = ", ".join(path.name for path in final_paths)
        raise FileExistsError(f"Final DocBench artifacts already exist ({names}); use --resume")


def _write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = output_dir / MANIFEST_FILE_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _validate_or_write_run_config(
    *,
    config: DocBenchGenerationConfig,
    selected_records: Sequence[DocBenchRecord],
    parameters: Mapping[str, Any],
) -> Path:
    path = config.output_dir / RUN_CONFIG_FILE_NAME
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_dir": str(config.data_dir.resolve()),
        "selection_sha256": _selection_sha256(selected_records),
        "selected_ids": [record.id for record in selected_records],
        "parameters": dict(parameters),
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read DocBench run config: {path}") from exc
        if not config.resume:
            raise FileExistsError(f"DocBench run config already exists; use --resume: {path}")
        if existing != payload:
            raise ValueError(
                "DocBench resume configuration or selected rows changed; "
                f"use the original command or a new output directory: {path}"
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _selection_sha256(records: Sequence[DocBenchRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record.source_payload(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _configure_logging(level: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[console, file_handler],
        force=True,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _require_unique(values: Sequence[Any], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
