from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from file_agent.chunking import Chunk, chunk_document
from file_agent.document import Document
from file_agent.llm.base import LLMClient
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
from file_agent.retrieval import SearchResult, SemanticModel, search_chunks


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    documents_count: int
    chunks_count: int


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    return [parse_file(file_path) for file_path in file_paths]


def chunk_documents(
    documents: list[Document],
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap=overlap,
            )
        )

    return chunks


def answer_files(
    file_paths: Iterable[str | Path],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    semantic_model: SemanticModel | None = None,
    use_semantic: bool = True,
) -> RAGResponse:
    documents = load_documents(file_paths)
    return answer_documents(
        documents=documents,
        question=question,
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        semantic_model=semantic_model,
        use_semantic=use_semantic,
    )


def answer_documents(
    documents: list[Document],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    semantic_model: SemanticModel | None = None,
    use_semantic: bool = True,
) -> RAGResponse:
    chunks = chunk_documents(
        documents=documents,
        max_chars=max_chars,
        overlap=overlap,
    )
    results = search_chunks(
        query=question,
        chunks=chunks,
        top_k=top_k,
        semantic_model=semantic_model,
        use_semantic=use_semantic,
    )
    answer = answer_question_with_context(
        question=question,
        results=results,
        llm_client=llm_client,
    )

    return RAGResponse(
        answer=answer,
        sources=results,
        documents_count=len(documents),
        chunks_count=len(chunks),
    )
