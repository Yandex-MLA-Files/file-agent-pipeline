import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from file_agent.chunking import Chunk, chunk_document, get_embedding_tokenizer
from file_agent.document import Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.base import LLMClient
from file_agent.pipeline import parse_file
from file_agent.planner import plan_subqueries
from file_agent.qa import answer_question_with_context
from file_agent.retrieval import Retriever, SearchResult
from file_agent.router import QueryType, classify_query
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    documents_count: int
    chunks_count: int
    query_type: QueryType | None = None


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    file_paths = list(file_paths)

    with tracer.start_as_current_span("file_agent.load_documents") as span:
        span.set_attribute("file_agent.file_count", len(file_paths))

        documents = [parse_file(file_path) for file_path in file_paths]

        logger.info("Loaded %d document(s)", len(documents))
        return documents


def chunk_documents(
    documents: list[Document],
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:

    tokenizer = get_embedding_tokenizer()
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap=overlap,
                tokenizer=tokenizer,
            )
        )

    return chunks


def index_documents(
    documents: list[Document],
    retriever: Retriever,
    max_chars: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    with tracer.start_as_current_span("file_agent.index_documents") as span:
        span.set_attribute("file_agent.document_count", len(documents))

        chunks = chunk_documents(
            documents=documents,
            max_chars=max_chars,
            overlap=overlap,
        )
        retriever.index(chunks)

        span.set_attribute("file_agent.chunk_count", len(chunks))
        logger.info("Indexed %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return chunks


def answer_indexed_documents(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
) -> RAGResponse:
    with tracer.start_as_current_span("file_agent.answer_indexed_documents") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.top_k", top_k)

        results = retriever.search(query=question, top_k=top_k)
        response = answer_with_results(
            question=question,
            results=results,
            llm_client=llm_client,
            documents_count=documents_count,
            chunks_count=chunks_count,
        )

        span.set_attribute("file_agent.result_count", len(results))
        return response


def answer_indexed_documents_with_routing(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
    router_llm_client: LLMClient | None = None,
) -> RAGResponse:
    active_router_llm_client = router_llm_client or llm_client
    with tracer.start_as_current_span("file_agent.answer_indexed_documents_with_routing") as span:
        query_type = classify_query(question, active_router_llm_client)
        span.set_attribute("file_agent.query_type", query_type.value)

        if query_type == QueryType.COMPLEX:
            response = answer_indexed_documents_with_plan(
                question=question,
                llm_client=llm_client,
                retriever=retriever,
                documents_count=documents_count,
                chunks_count=chunks_count,
                top_k=top_k,
                router_llm_client=active_router_llm_client,
            )
        else:
            response = answer_indexed_documents(
                question=question,
                llm_client=llm_client,
                retriever=retriever,
                documents_count=documents_count,
                chunks_count=chunks_count,
                top_k=top_k,
            )
        response.query_type = query_type
        return response


def answer_indexed_documents_with_plan(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    documents_count: int,
    chunks_count: int,
    top_k: int = 5,
    router_llm_client: LLMClient | None = None,
) -> RAGResponse:
    active_router_llm_client = router_llm_client or llm_client
    with tracer.start_as_current_span("file_agent.answer_indexed_documents_with_plan") as span:
        span.set_attribute("file_agent.question", question)

        subqueries = plan_subqueries(question, active_router_llm_client)
        span.set_attribute("file_agent.subquery_count", len(subqueries))

        results_by_subquery = [
            retriever.search(query=subquery, top_k=top_k) for subquery in subqueries
        ]
        results = merge_search_results(results_by_subquery, top_k=top_k)
        span.set_attribute("file_agent.result_count", len(results))

        return answer_with_results(
            question=question,
            results=results,
            llm_client=llm_client,
            documents_count=documents_count,
            chunks_count=chunks_count,
        )


def merge_search_results(
    results_by_subquery: list[list[SearchResult]],
    top_k: int,
) -> list[SearchResult]:
    """Combine per-subquery search results into one ranked, deduplicated list.

    The same chunk can be found by several subqueries; keep its best score so
    the merged context stays bounded to top_k regardless of subquery count.
    """
    best_by_chunk_id: dict[str, SearchResult] = {}
    for results in results_by_subquery:
        for result in results:
            existing = best_by_chunk_id.get(result.chunk.id)
            if existing is None or result.score > existing.score:
                best_by_chunk_id[result.chunk.id] = result

    merged = sorted(best_by_chunk_id.values(), key=lambda result: result.score, reverse=True)
    return merged[:top_k]


def answer_with_results(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
    documents_count: int,
    chunks_count: int,
) -> RAGResponse:
    answer = answer_question_with_context(
        question=question,
        results=results,
        llm_client=llm_client,
    )

    return RAGResponse(
        answer=answer,
        sources=results,
        documents_count=documents_count,
        chunks_count=chunks_count,
    )


def answer_files(
    file_paths: Iterable[str | Path],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
) -> RAGResponse:
    documents = load_documents(file_paths)
    return answer_documents(
        documents=documents,
        question=question,
        llm_client=llm_client,
        top_k=top_k,
        max_chars=max_chars,
        overlap=overlap,
        retriever=retriever,
    )


def answer_documents(
    documents: list[Document],
    question: str,
    llm_client: LLMClient,
    top_k: int = 5,
    max_chars: int = 1000,
    overlap: int = 100,
    retriever: Retriever | None = None,
) -> RAGResponse:
    active_retriever = retriever or LanceDBRetriever()
    chunks = index_documents(
        documents=documents,
        retriever=active_retriever,
        max_chars=max_chars,
        overlap=overlap,
    )
    return answer_indexed_documents(
        question=question,
        llm_client=llm_client,
        retriever=active_retriever,
        documents_count=len(documents),
        chunks_count=len(chunks),
        top_k=top_k,
    )
