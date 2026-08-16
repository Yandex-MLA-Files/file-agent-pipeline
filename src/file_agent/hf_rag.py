import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from file_agent.document import Document
from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.hf_dataset import QADatasetRecord, download_record_documents
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.qa import select_context_passages
from file_agent.rag import (
    RAGMode,
    answer_indexed_documents,
    ingest_documents,
    load_documents,
    resolve_max_tool_rounds,
    resolve_rag_mode,
)
from file_agent.retrieval import Retriever, SearchResult
from file_agent.vlm.base import VLMClient

LLM_CONTEXT_METADATA_KEY = "_llm_context"

DocumentLoader = Callable[[list[str | Path]], list[Document]]


@dataclass(frozen=True)
class RetrievedContext:
    rank: int
    chunk_id: str
    document_id: str
    text: str
    retrieval_text: str
    score: float
    metadata_json: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievedContext":
        rank = value.get("rank")
        score = value.get("score")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("context rank must be a positive integer")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("context score must be a number")

        strings: dict[str, str] = {}
        for field_name in (
            "chunk_id",
            "document_id",
            "text",
            "retrieval_text",
            "metadata_json",
        ):
            field_value = value.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"context {field_name} must be a non-empty string")
            strings[field_name] = field_value

        try:
            metadata = json.loads(strings["metadata_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("context metadata_json must contain valid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("context metadata_json must contain a JSON object")

        return cls(
            rank=rank,
            chunk_id=strings["chunk_id"],
            document_id=strings["document_id"],
            text=strings["text"],
            retrieval_text=strings["retrieval_text"],
            score=float(score),
            metadata_json=strings["metadata_json"],
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "score": self.score,
            "metadata_json": self.metadata_json,
        }


@dataclass(frozen=True)
class GeneratedQARecord:
    id: str
    question: str
    doc_ids: tuple[str, ...]
    answer_model: str
    contexts: tuple[RetrievedContext, ...]
    answer: str
    rag_mode: str = "standard"
    stop_reason: str = "answer_generated"
    search_queries: tuple[str, ...] = ()
    tool_calls_json: str = "[]"
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedQARecord":
        source_record = QADatasetRecord.from_row(value)
        answer_model = value.get("answer_model")
        if not isinstance(answer_model, str) or not answer_model.strip():
            raise ValueError("answer_model must be a non-empty string")

        raw_contexts = value.get("contexts")
        if not isinstance(raw_contexts, Sequence) or isinstance(raw_contexts, (str, bytes)):
            raise ValueError("contexts must be a list")

        contexts: list[RetrievedContext] = []
        for raw_context in raw_contexts:
            if not isinstance(raw_context, Mapping):
                raise ValueError("contexts must contain objects")
            contexts.append(RetrievedContext.from_dict(raw_context))

        expected_ranks = list(range(1, len(contexts) + 1))
        if [context.rank for context in contexts] != expected_ranks:
            raise ValueError("context ranks must be consecutive and start at 1")

        rag_mode = resolve_rag_mode(str(value.get("rag_mode", "standard")))
        stop_reason = value.get("stop_reason", "answer_generated")
        if not isinstance(stop_reason, str) or not stop_reason.strip():
            raise ValueError("stop_reason must be a non-empty string")

        raw_search_queries = value.get("search_queries", [])
        if not isinstance(raw_search_queries, Sequence) or isinstance(
            raw_search_queries, (str, bytes)
        ):
            raise ValueError("search_queries must be a list")
        search_queries = tuple(raw_search_queries)
        if any(not isinstance(query, str) or not query.strip() for query in search_queries):
            raise ValueError("search_queries must contain only non-empty strings")

        tool_calls_json = value.get("tool_calls_json", "[]")
        if not isinstance(tool_calls_json, str):
            raise ValueError("tool_calls_json must be a string")
        try:
            tool_calls = json.loads(tool_calls_json)
        except json.JSONDecodeError as exc:
            raise ValueError("tool_calls_json must contain valid JSON") from exc
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls_json must contain a JSON list")

        counters = {
            name: _non_negative_int(value.get(name, 0), name)
            for name in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        duration_seconds = value.get("duration_seconds", 0.0)
        if (
            not isinstance(duration_seconds, (int, float))
            or isinstance(duration_seconds, bool)
            or not math.isfinite(duration_seconds)
            or duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be a non-negative number")

        return cls(
            id=source_record.id,
            question=source_record.question,
            doc_ids=source_record.doc_ids,
            answer_model=answer_model,
            contexts=tuple(contexts),
            answer=source_record.answer,
            rag_mode=rag_mode,
            stop_reason=stop_reason,
            search_queries=search_queries,
            tool_calls_json=tool_calls_json,
            duration_seconds=float(duration_seconds),
            **counters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "doc_ids": list(self.doc_ids),
            "answer_model": self.answer_model,
            "contexts": [context.to_dict() for context in self.contexts],
            "answer": self.answer,
            "rag_mode": self.rag_mode,
            "stop_reason": self.stop_reason,
            "search_queries": list(self.search_queries),
            "tool_calls_json": self.tool_calls_json,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
        }


def process_hf_qa_record(
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
    rag_mode: str | None = None,
    max_tool_rounds: int | None = None,
    vlm_client: VLMClient | None = None,
) -> GeneratedQARecord:
    document_paths = download_record_documents(
        record=record,
        dataset_id=dataset_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return process_qa_record(
        record=record,
        document_paths=document_paths,
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        retriever=retriever,
        document_loader=document_loader,
        rag_mode=rag_mode,
        max_tool_rounds=max_tool_rounds,
        vlm_client=vlm_client,
    )


def process_qa_record(
    record: QADatasetRecord,
    document_paths: list[str | Path],
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
    document_loader: DocumentLoader | None = None,
    rag_mode: str | None = None,
    max_tool_rounds: int | None = None,
    vlm_client: VLMClient | None = None,
) -> GeneratedQARecord:
    if len(document_paths) != len(record.doc_ids):
        raise ValueError("document_paths count must match record.doc_ids count")

    active_mode: RAGMode = resolve_rag_mode(rag_mode)
    active_max_tool_rounds = (
        resolve_max_tool_rounds(max_tool_rounds) if active_mode == "tool_agent" else None
    )
    active_document_loader = document_loader or load_documents
    documents = active_document_loader(document_paths)
    for document, doc_id in zip(documents, record.doc_ids, strict=True):
        for block in document.blocks:
            block.metadata["dataset_record_id"] = record.id
            block.metadata["dataset_doc_id"] = doc_id

    active_retriever = retriever if retriever is not None else LanceDBRetriever()
    try:
        chunks = ingest_documents(
            documents=documents,
            retriever=active_retriever,
            max_chars=max_chars,
            overlap=overlap,
        )
        counters_before = _llm_counters(llm_client)
        started_at = time.perf_counter()
        asset_store = (
            InMemoryDocumentAssetStore.from_files(document_paths)
            if active_mode == "tool_agent" and vlm_client is not None
            else None
        )
        response = answer_indexed_documents(
            question=record.question,
            llm_client=llm_client,
            retriever=active_retriever,
            documents_count=len(documents),
            chunks_count=len(chunks),
            top_k=top_k,
            documents=documents,
            mode=active_mode,
            max_tool_rounds=active_max_tool_rounds,
            vlm_client=vlm_client,
            asset_store=asset_store,
            require_evidence_tool=active_mode == "tool_agent",
            required_evidence_files=(
                tuple(document.file_name for document in documents)
                if active_mode == "tool_agent" and len(documents) > 1
                else ()
            ),
        )
        duration_seconds = time.perf_counter() - started_at
        counters_after = _llm_counters(llm_client)
        usage = {
            name: max(0, counters_after[name] - counters_before[name]) for name in counters_before
        }
        contexts = serialize_search_results(response.sources)

        return GeneratedQARecord(
            id=record.id,
            question=record.question,
            doc_ids=record.doc_ids,
            answer_model=response.answer,
            contexts=contexts,
            answer=record.answer,
            rag_mode=active_mode,
            stop_reason=response.stop_reason,
            search_queries=tuple(response.search_queries),
            tool_calls_json=json.dumps(
                response.tool_calls,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            duration_seconds=duration_seconds,
            **usage,
        )
    finally:
        active_retriever.clear()


def serialize_search_results(
    results: list[SearchResult],
) -> tuple[RetrievedContext, ...]:
    contexts: list[RetrievedContext] = []

    for rank, (result, passage) in enumerate(select_context_passages(results), start=1):
        metadata = dict(result.chunk.metadata)
        llm_context = metadata.pop(LLM_CONTEXT_METADATA_KEY, None)
        parent_context = metadata.pop("context", None)
        passage = str(llm_context or parent_context or passage)
        contexts.append(
            RetrievedContext(
                rank=rank,
                chunk_id=result.chunk.id,
                document_id=str(metadata.get("dataset_doc_id", "")),
                text=passage,
                retrieval_text=result.chunk.text,
                score=float(result.score),
                metadata_json=json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
        )

    return tuple(contexts)


def _llm_counters(llm_client: LLMClient) -> dict[str, int]:
    return {
        "llm_calls": _counter_value(llm_client, "request_count"),
        "prompt_tokens": _counter_value(llm_client, "prompt_tokens"),
        "completion_tokens": _counter_value(llm_client, "completion_tokens"),
        "total_tokens": _counter_value(llm_client, "total_tokens"),
    }


def _counter_value(component: object, name: str) -> int:
    value = getattr(component, name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value
