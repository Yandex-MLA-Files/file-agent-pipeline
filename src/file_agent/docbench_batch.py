import hashlib
import json
import logging
import math
import os
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from file_agent.docbench_dataset import DocBenchRecord
from file_agent.document import Document
from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.hf_batch import RAG_PIPELINE_VERSION, build_generation_parameters
from file_agent.hf_rag import RetrievedContext, serialize_search_results
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.pipeline import parse_file
from file_agent.rag import (
    answer_indexed_documents,
    ingest_documents,
    resolve_max_tool_rounds,
    resolve_rag_mode,
)
from file_agent.retrieval import Retriever
from file_agent.vlm.base import VLMClient

CHECKPOINT_SCHEMA_VERSION = 1
DOCBENCH_RUNNER_VERSION = "docbench-local-v1"
CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
FAILURES_DIRECTORY_NAME = "failures"
DOCUMENTS_DIRECTORY_NAME = "documents"
LOGGER = logging.getLogger(__name__)

DocumentLoader = Callable[[Path, VLMClient | None], Document]
RetrieverFactory = Callable[[], Retriever]
AssetStoreFactory = Callable[[list[str | Path]], InMemoryDocumentAssetStore]


@dataclass(frozen=True)
class DocBenchGeneratedRecord:
    id: str
    folder_id: int
    question_index: int
    question: str
    reference_answer: str
    question_type: str
    evidence: str
    domain: str
    source_file: str
    answer_model: str
    contexts: tuple[RetrievedContext, ...]
    rag_mode: str
    stop_reason: str
    search_queries: tuple[str, ...]
    tool_calls_json: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    vlm_calls: int
    vlm_prompt_tokens: int
    vlm_completion_tokens: int
    vlm_total_tokens: int
    document_vlm_calls: int
    document_vlm_prompt_tokens: int
    document_vlm_completion_tokens: int
    document_vlm_total_tokens: int
    answer_duration_seconds: float
    document_preparation_seconds: float
    amortized_duration_seconds: float
    document_blocks: int
    document_chunks: int
    document_pages: int

    @property
    def question_number(self) -> int:
        return self.question_index + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "folder_id": self.folder_id,
            "question_index": self.question_index,
            "question_number": self.question_number,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "question_type": self.question_type,
            "evidence": self.evidence,
            "domain": self.domain,
            "source_file": self.source_file,
            "answer_model": self.answer_model,
            "contexts": [context.to_dict() for context in self.contexts],
            "rag_mode": self.rag_mode,
            "stop_reason": self.stop_reason,
            "search_queries": list(self.search_queries),
            "tool_calls_json": self.tool_calls_json,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "vlm_calls": self.vlm_calls,
            "vlm_prompt_tokens": self.vlm_prompt_tokens,
            "vlm_completion_tokens": self.vlm_completion_tokens,
            "vlm_total_tokens": self.vlm_total_tokens,
            "document_vlm_calls": self.document_vlm_calls,
            "document_vlm_prompt_tokens": self.document_vlm_prompt_tokens,
            "document_vlm_completion_tokens": self.document_vlm_completion_tokens,
            "document_vlm_total_tokens": self.document_vlm_total_tokens,
            "answer_duration_seconds": self.answer_duration_seconds,
            "document_preparation_seconds": self.document_preparation_seconds,
            "amortized_duration_seconds": self.amortized_duration_seconds,
            "document_blocks": self.document_blocks,
            "document_chunks": self.document_chunks,
            "document_pages": self.document_pages,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocBenchGeneratedRecord":
        raw_contexts = value.get("contexts")
        if not isinstance(raw_contexts, Sequence) or isinstance(raw_contexts, (str, bytes)):
            raise ValueError("DocBench checkpoint contexts must be a list")
        contexts = tuple(
            RetrievedContext.from_dict(context)
            for context in raw_contexts
            if isinstance(context, Mapping)
        )
        if len(contexts) != len(raw_contexts):
            raise ValueError("DocBench checkpoint contexts must contain objects")
        if [context.rank for context in contexts] != list(range(1, len(contexts) + 1)):
            raise ValueError("DocBench checkpoint context ranks must start at 1 and be consecutive")

        string_fields = {
            name: _required_string(value.get(name), name)
            for name in (
                "id",
                "question",
                "reference_answer",
                "question_type",
                "evidence",
                "domain",
                "source_file",
                "answer_model",
                "rag_mode",
                "stop_reason",
                "tool_calls_json",
            )
        }
        raw_queries = value.get("search_queries")
        if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, (str, bytes)):
            raise ValueError("DocBench checkpoint search_queries must be a list")
        queries = tuple(_required_string(query, "search query") for query in raw_queries)
        try:
            tool_calls = json.loads(string_fields["tool_calls_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("DocBench checkpoint tool_calls_json must be valid JSON") from exc
        if not isinstance(tool_calls, list):
            raise ValueError("DocBench checkpoint tool_calls_json must contain a list")

        integer_fields = {
            name: _non_negative_int(value.get(name), name)
            for name in (
                "folder_id",
                "question_index",
                "llm_calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "vlm_calls",
                "vlm_prompt_tokens",
                "vlm_completion_tokens",
                "vlm_total_tokens",
                "document_vlm_calls",
                "document_vlm_prompt_tokens",
                "document_vlm_completion_tokens",
                "document_vlm_total_tokens",
                "document_blocks",
                "document_chunks",
                "document_pages",
            )
        }
        duration_fields = {
            name: _non_negative_float(value.get(name), name)
            for name in (
                "answer_duration_seconds",
                "document_preparation_seconds",
                "amortized_duration_seconds",
            )
        }
        return cls(
            contexts=contexts,
            search_queries=queries,
            **string_fields,
            **integer_fields,
            **duration_fields,
        )


@dataclass(frozen=True)
class DocumentPreparationResult:
    folder_id: int
    source_file: str
    source_sha256: str
    source_size_bytes: int
    block_count: int
    chunk_count: int
    page_count: int
    selected_question_count: int
    parse_and_index_seconds: float
    vlm_calls: int
    vlm_prompt_tokens: int
    vlm_completion_tokens: int
    vlm_total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DocBenchFailure:
    id: str
    folder_id: int
    error_type: str
    message: str
    failure_path: Path


@dataclass(frozen=True)
class DocBenchBatchResult:
    records: tuple[DocBenchGeneratedRecord, ...]
    document_preparations: tuple[DocumentPreparationResult, ...]
    failures: tuple[DocBenchFailure, ...]
    processed_count: int
    resumed_count: int

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def is_complete(self) -> bool:
        return not self.failures


def generate_docbench_records(
    records: Sequence[DocBenchRecord],
    *,
    llm_client: LLMClient,
    output_dir: str | Path,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    resume: bool = False,
    rag_mode: str = "tool_agent",
    max_tool_rounds: int | None = None,
    vlm_client: VLMClient | None = None,
    fail_fast: bool = False,
    document_loader: DocumentLoader | None = None,
    retriever_factory: RetrieverFactory | None = None,
    asset_store_factory: AssetStoreFactory | None = None,
) -> DocBenchBatchResult:
    """Generate DocBench answers with one parse/index operation per source PDF."""
    if not records:
        raise ValueError("DocBench records must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")
    active_rag_mode = resolve_rag_mode(rag_mode)
    active_max_tool_rounds = (
        resolve_max_tool_rounds(max_tool_rounds) if active_rag_mode == "tool_agent" else None
    )

    output_path = Path(output_dir)
    checkpoints_dir = output_path / CHECKPOINTS_DIRECTORY_NAME
    failures_dir = output_path / FAILURES_DIRECTORY_NAME
    documents_dir = output_path / DOCUMENTS_DIRECTORY_NAME
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    parameters = build_docbench_generation_parameters(
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        rag_mode=active_rag_mode,
        max_tool_rounds=active_max_tool_rounds,
    )
    _validate_checkpoint_policy(records, checkpoints_dir, resume)

    load_document = document_loader or _load_document
    create_retriever = retriever_factory or LanceDBRetriever
    create_asset_store = asset_store_factory or InMemoryDocumentAssetStore.from_files
    grouped = _group_by_document(records)
    generated_by_id: dict[str, DocBenchGeneratedRecord] = {}
    preparations: list[DocumentPreparationResult] = []
    failures: list[DocBenchFailure] = []
    processed_count = 0
    resumed_count = 0
    source_hashes: dict[Path, str] = {}

    for folder_id, document_records in grouped:
        pending: list[DocBenchRecord] = []
        for record in document_records:
            checkpoint_path = _checkpoint_path(checkpoints_dir, record.id)
            if checkpoint_path.exists():
                generated = _load_checkpoint(
                    checkpoint_path,
                    record,
                    parameters,
                    _source_sha256(record, source_hashes),
                )
                generated_by_id[record.id] = generated
                resumed_count += 1
                LOGGER.info("Loaded DocBench checkpoint %s", record.id)
            else:
                pending.append(record)

        if not pending:
            continue

        pdf_path = document_records[0].pdf_path
        retriever = create_retriever()
        asset_store: InMemoryDocumentAssetStore | None = None
        prep_started = time.perf_counter()
        vlm_before_prep = _component_counters(vlm_client, "vlm")
        try:
            document = load_document(pdf_path, vlm_client)
            dataset_doc_id = f"{folder_id}/{pdf_path.name}"
            for block in document.blocks:
                block.metadata["dataset_doc_id"] = dataset_doc_id
            chunks = ingest_documents(
                documents=[document],
                retriever=retriever,
                max_chars=max_chars,
                overlap=overlap,
            )
            prep_seconds = time.perf_counter() - prep_started
            prep_vlm = _counter_delta(vlm_before_prep, _component_counters(vlm_client, "vlm"))
            preparation = DocumentPreparationResult(
                folder_id=folder_id,
                source_file=pdf_path.name,
                source_sha256=_file_sha256_cached(pdf_path, source_hashes),
                source_size_bytes=pdf_path.stat().st_size,
                block_count=len(document.blocks),
                chunk_count=len(chunks),
                page_count=_page_count(document),
                selected_question_count=len(document_records),
                parse_and_index_seconds=prep_seconds,
                **prep_vlm,
            )
            preparations.append(preparation)
            _write_json_atomic(
                documents_dir / f"{folder_id:03d}.json",
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "created_at_utc": _utc_timestamp(),
                    "parameters": parameters,
                    "document": preparation.to_dict(),
                },
            )

            if active_rag_mode == "tool_agent" and vlm_client is not None:
                asset_store = create_asset_store([pdf_path])

            for record in pending:
                try:
                    generated = _answer_record(
                        record=record,
                        document=document,
                        document_preparation=preparation,
                        llm_client=llm_client,
                        vlm_client=vlm_client,
                        retriever=retriever,
                        chunks_count=len(chunks),
                        top_k=top_k,
                        rag_mode=active_rag_mode,
                        max_tool_rounds=active_max_tool_rounds,
                        asset_store=asset_store,
                    )
                    _write_checkpoint(
                        _checkpoint_path(checkpoints_dir, record.id),
                        record,
                        parameters,
                        _source_sha256(record, source_hashes),
                        generated,
                    )
                    failure_path = _failure_path(failures_dir, record.id)
                    if failure_path.exists():
                        failure_path.unlink()
                    generated_by_id[record.id] = generated
                    processed_count += 1
                    LOGGER.info(
                        "Processed DocBench question %s (%s/%s total complete)",
                        record.id,
                        len(generated_by_id),
                        len(records),
                    )
                except Exception as exc:
                    failure = _write_failure(failures_dir, record, parameters, exc)
                    failures.append(failure)
                    LOGGER.exception("DocBench question failed: %s", record.id)
                    if fail_fast:
                        raise
        except Exception as exc:
            LOGGER.exception("DocBench document preparation failed: folder %s", folder_id)
            for record in pending:
                if record.id in generated_by_id or any(item.id == record.id for item in failures):
                    continue
                failures.append(_write_failure(failures_dir, record, parameters, exc))
            if fail_fast:
                raise
        finally:
            if asset_store is not None:
                asset_store.clear()
            retriever.clear()

    ordered_records = tuple(
        generated_by_id[record.id] for record in records if record.id in generated_by_id
    )
    return DocBenchBatchResult(
        records=ordered_records,
        document_preparations=tuple(preparations),
        failures=tuple(failures),
        processed_count=processed_count,
        resumed_count=resumed_count,
    )


def build_docbench_generation_parameters(
    *,
    llm_client: LLMClient,
    top_k: int,
    max_chars: int,
    overlap: int,
    rag_mode: str,
    max_tool_rounds: int | None,
) -> dict[str, Any]:
    parameters = build_generation_parameters(
        dataset_id="DocBench/local",
        revision=None,
        llm_client=llm_client,
        retriever=None,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        rag_mode=rag_mode,
        max_tool_rounds=max_tool_rounds,
    )
    parameters.update(
        {
            "docbench_runner_version": DOCBENCH_RUNNER_VERSION,
            "rag_pipeline_version": RAG_PIPELINE_VERSION,
            "document_reuse": "parse-and-index-once-per-pdf",
            "vlm_max_tokens": os.getenv("VLM_MAX_TOKENS", "1000").strip(),
            "vlm_temperature": os.getenv("VLM_TEMPERATURE", "0.2").strip(),
            "vlm_enable_thinking": os.getenv("VLM_ENABLE_THINKING", "").strip(),
            "vlm_max_retries": os.getenv("VLM_MAX_RETRIES", "0").strip(),
            "vlm_max_figures": os.getenv("VLM_MAX_FIGURES", "8").strip(),
            "vlm_min_figure_area": os.getenv("VLM_MIN_FIGURE_AREA", "5000").strip(),
            "vlm_max_visual_pixels": os.getenv("VLM_MAX_VISUAL_PIXELS", "1500000").strip(),
        }
    )
    return parameters


def _answer_record(
    *,
    record: DocBenchRecord,
    document: Document,
    document_preparation: DocumentPreparationResult,
    llm_client: LLMClient,
    vlm_client: VLMClient | None,
    retriever: Retriever,
    chunks_count: int,
    top_k: int,
    rag_mode: str,
    max_tool_rounds: int | None,
    asset_store: InMemoryDocumentAssetStore | None,
) -> DocBenchGeneratedRecord:
    llm_before = _component_counters(llm_client, "llm")
    vlm_before = _component_counters(vlm_client, "vlm")
    started_at = time.perf_counter()
    response = answer_indexed_documents(
        question=record.question,
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=chunks_count,
        top_k=top_k,
        documents=[document],
        mode=rag_mode,
        max_tool_rounds=max_tool_rounds,
        vlm_client=vlm_client,
        asset_store=asset_store,
        require_evidence_tool=rag_mode == "tool_agent",
    )
    duration_seconds = time.perf_counter() - started_at
    llm_usage = _counter_delta(llm_before, _component_counters(llm_client, "llm"))
    vlm_usage = _counter_delta(vlm_before, _component_counters(vlm_client, "vlm"))
    answer = response.answer.strip()
    if not answer:
        raise ValueError("RAG returned an empty answer")

    return DocBenchGeneratedRecord(
        id=record.id,
        folder_id=record.folder_id,
        question_index=record.question_index,
        question=record.question,
        reference_answer=record.reference_answer,
        question_type=record.question_type,
        evidence=record.evidence,
        domain=record.domain,
        source_file=record.source_file,
        answer_model=answer,
        contexts=serialize_search_results(response.sources),
        rag_mode=rag_mode,
        stop_reason=response.stop_reason,
        search_queries=tuple(response.search_queries),
        tool_calls_json=json.dumps(
            response.tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        answer_duration_seconds=duration_seconds,
        document_preparation_seconds=document_preparation.parse_and_index_seconds,
        amortized_duration_seconds=(
            duration_seconds
            + document_preparation.parse_and_index_seconds
            / document_preparation.selected_question_count
        ),
        document_blocks=document_preparation.block_count,
        document_chunks=document_preparation.chunk_count,
        document_pages=document_preparation.page_count,
        document_vlm_calls=document_preparation.vlm_calls,
        document_vlm_prompt_tokens=document_preparation.vlm_prompt_tokens,
        document_vlm_completion_tokens=document_preparation.vlm_completion_tokens,
        document_vlm_total_tokens=document_preparation.vlm_total_tokens,
        **llm_usage,
        **vlm_usage,
    )


def _load_document(path: Path, vlm_client: VLMClient | None) -> Document:
    return parse_file(path, vlm_client=vlm_client)


def _group_by_document(
    records: Sequence[DocBenchRecord],
) -> tuple[tuple[int, tuple[DocBenchRecord, ...]], ...]:
    grouped: dict[int, list[DocBenchRecord]] = defaultdict(list)
    paths: dict[int, Path] = {}
    for record in records:
        previous_path = paths.setdefault(record.folder_id, record.pdf_path)
        if previous_path.resolve() != record.pdf_path.resolve():
            raise ValueError(f"DocBench folder {record.folder_id} points to multiple PDFs")
        grouped[record.folder_id].append(record)
    return tuple((folder_id, tuple(values)) for folder_id, values in grouped.items())


def _validate_checkpoint_policy(
    records: Sequence[DocBenchRecord],
    checkpoints_dir: Path,
    resume: bool,
) -> None:
    existing = sorted(checkpoints_dir.glob("*.json"))
    if existing and not resume:
        raise FileExistsError(
            f"DocBench checkpoints already exist in {checkpoints_dir}; use --resume or a new output"
        )


def _write_checkpoint(
    path: Path,
    source_record: DocBenchRecord,
    parameters: Mapping[str, Any],
    source_sha256: str,
    generated_record: DocBenchGeneratedRecord,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at_utc": _utc_timestamp(),
            "parameters": dict(parameters),
            "source_sha256": source_sha256,
            "source": source_record.source_payload(),
            "result": generated_record.to_dict(),
        },
    )


def _load_checkpoint(
    path: Path,
    source_record: DocBenchRecord,
    expected_parameters: Mapping[str, Any],
    expected_source_sha256: str,
) -> DocBenchGeneratedRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read DocBench checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported DocBench checkpoint: {path}")
    if payload.get("parameters") != dict(expected_parameters):
        raise ValueError(f"DocBench checkpoint parameters do not match this run: {path}")
    if payload.get("source_sha256") != expected_source_sha256:
        raise ValueError(f"DocBench source PDF changed since checkpoint: {path}")
    if payload.get("source") != source_record.source_payload():
        raise ValueError(f"DocBench source question changed since checkpoint: {path}")
    raw_result = payload.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError(f"DocBench checkpoint has no result object: {path}")
    result = DocBenchGeneratedRecord.from_dict(raw_result)
    _validate_result_matches_source(result, source_record)
    if result.rag_mode != expected_parameters.get("rag_mode"):
        raise ValueError(f"DocBench checkpoint RAG mode does not match this run: {path}")
    return result


def _validate_result_matches_source(
    result: DocBenchGeneratedRecord,
    source: DocBenchRecord,
) -> None:
    expected = source.source_payload()
    actual = {
        "id": result.id,
        "folder_id": result.folder_id,
        "question_index": result.question_index,
        "question": result.question,
        "reference_answer": result.reference_answer,
        "question_type": result.question_type,
        "evidence": result.evidence,
        "domain": result.domain,
        "source_file": result.source_file,
    }
    if actual != expected:
        raise ValueError(f"DocBench checkpoint result does not match source: {source.id}")


def _write_failure(
    failures_dir: Path,
    record: DocBenchRecord,
    parameters: Mapping[str, Any],
    exc: Exception,
) -> DocBenchFailure:
    path = _failure_path(failures_dir, record.id)
    _write_json_atomic(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "failed_at_utc": _utc_timestamp(),
            "parameters": dict(parameters),
            "source": record.source_payload(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return DocBenchFailure(
        id=record.id,
        folder_id=record.folder_id,
        error_type=type(exc).__name__,
        message=str(exc),
        failure_path=path,
    )


def _checkpoint_path(directory: Path, record_id: str) -> Path:
    return directory / f"{record_id}.json"


def _failure_path(directory: Path, record_id: str) -> Path:
    return directory / f"{record_id}.json"


def _source_sha256(record: DocBenchRecord, cache: dict[Path, str]) -> str:
    payload = {
        "pdf_sha256": _file_sha256_cached(record.pdf_path, cache),
        "record": record.source_payload(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256_cached(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[resolved] = digest.hexdigest()
    return cache[resolved]


def _component_counters(component: object | None, prefix: str) -> dict[str, int]:
    token_prefix = "" if prefix == "llm" else f"{prefix}_"
    return {
        f"{prefix}_calls": _counter_value(component, "request_count"),
        f"{token_prefix}prompt_tokens": _counter_value(component, "prompt_tokens"),
        f"{token_prefix}completion_tokens": _counter_value(component, "completion_tokens"),
        f"{token_prefix}total_tokens": _counter_value(component, "total_tokens"),
    }


def _counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {name: max(0, after[name] - value) for name, value in before.items()}


def _counter_value(component: object | None, name: str) -> int:
    value = getattr(component, name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _page_count(document: Document) -> int:
    value = document.metadata.get("total_pages", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    # Reference/evidence may legitimately be empty in a custom DocBench subset.
    if field_name not in {"reference_answer", "evidence"} and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _non_negative_float(value: Any, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)
