import json
from dataclasses import dataclass
from pathlib import Path

from file_agent.hf_dataset import QADatasetRecord, download_record_documents
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.rag import answer_indexed_documents, index_documents, load_documents
from file_agent.retrieval import Retriever, SearchResult


@dataclass(frozen=True)
class RetrievedContext:
    rank: int
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata_json: str

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
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

    def to_dict(self) -> dict:
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
    )


def process_qa_record(
    record: QADatasetRecord,
    document_paths: list[str | Path],
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
) -> GeneratedQARecord:
    if len(document_paths) != len(record.doc_ids):
        raise ValueError("document_paths count must match record.doc_ids count")

    documents = load_documents(document_paths)
    for document, doc_id in zip(documents, record.doc_ids, strict=True):
        for block in document.blocks:
            block.metadata["dataset_record_id"] = record.id
            block.metadata["dataset_doc_id"] = doc_id

    active_retriever = retriever if retriever is not None else LanceDBRetriever()
    try:
        chunks = index_documents(
            documents=documents,
            retriever=active_retriever,
            max_chars=max_chars,
            overlap=overlap,
        )
        response = answer_indexed_documents(
            question=record.question,
            llm_client=llm_client,
            retriever=active_retriever,
            documents_count=len(documents),
            chunks_count=len(chunks),
            top_k=top_k,
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


def serialize_search_results(
    results: list[SearchResult],
) -> tuple[RetrievedContext, ...]:
    contexts: list[RetrievedContext] = []

    for rank, result in enumerate(results, start=1):
        metadata = result.chunk.metadata
        contexts.append(
            RetrievedContext(
                rank=rank,
                chunk_id=result.chunk.id,
                document_id=str(metadata.get("dataset_doc_id", "")),
                text=result.chunk.text,
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
