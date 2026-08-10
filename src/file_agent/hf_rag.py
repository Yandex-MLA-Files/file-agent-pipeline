import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from file_agent.agent.loop import MAX_ITERATIONS_DEFAULT, run_react_agent
from file_agent.agent.observability import finish_trace, pipeline_trace
from file_agent.agent.tools import build_default_tools
from file_agent.document import Document
from file_agent.hf_dataset import QADatasetRecord, download_record_documents
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.qa import select_context_passages
from file_agent.rag import index_documents, load_documents
from file_agent.retrieval import Retriever, SearchResult

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

        return cls(
            id=source_record.id,
            question=source_record.question,
            doc_ids=source_record.doc_ids,
            answer_model=answer_model,
            contexts=tuple(contexts),
            answer=source_record.answer,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "doc_ids": list(self.doc_ids),
            "answer_model": self.answer_model,
            "contexts": [context.to_dict() for context in self.contexts],
            "answer": self.answer,
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
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
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
        max_iterations=max_iterations,
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
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> GeneratedQARecord:
    if len(document_paths) != len(record.doc_ids):
        raise ValueError("document_paths count must match record.doc_ids count")

    with pipeline_trace(record.question):
        active_document_loader = document_loader or load_documents
        documents = active_document_loader(document_paths)
        for document, doc_id in zip(documents, record.doc_ids, strict=True):
            for block in document.blocks:
                block.metadata["dataset_record_id"] = record.id
                block.metadata["dataset_doc_id"] = doc_id

        active_retriever = retriever if retriever is not None else LanceDBRetriever()
        try:
            index_documents(
                documents=documents,
                retriever=active_retriever,
                max_chars=max_chars,
                overlap=overlap,
            )
            document_path_map = {Path(path).name: Path(path) for path in document_paths}
            tools = build_default_tools(
                active_retriever, document_paths=document_path_map, default_top_k=top_k
            )
            response = run_react_agent(
                question=record.question,
                llm_client=llm_client,
                tools=tools,
                max_iterations=max_iterations,
            )
            contexts = serialize_search_results(response.sources)
            finish_trace(output=response.answer)

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


def serialize_search_results(
    results: list[SearchResult],
) -> tuple[RetrievedContext, ...]:
    contexts: list[RetrievedContext] = []

    for rank, (result, passage) in enumerate(select_context_passages(results), start=1):
        metadata = dict(result.chunk.metadata)
        metadata.pop("context", None)
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
