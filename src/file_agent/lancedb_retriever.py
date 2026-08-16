import json
import logging
import os
from functools import lru_cache
from typing import Protocol

import lancedb
import numpy as np
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FTS_LANGUAGE = "Russian"
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_MIN_SCORE = 0.25
DEFAULT_TABLE_NAME = "chunks"


class EmbeddingModel(Protocol):
    def encode(self, sentences): ...


class LanceDBRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        embedding_model_name: str | None = None,
        uri: str = "memory://",
        table_name: str = DEFAULT_TABLE_NAME,
        fts_language: str = DEFAULT_FTS_LANGUAGE,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
    ) -> None:
        if not -1.0 <= semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between -1 and 1")

        self._embedding_model = embedding_model
        self._embedding_model_name = resolve_semantic_model_name(embedding_model_name)
        self._connection = lancedb.connect(uri)
        self._table_name = table_name
        self._fts_language = fts_language
        self._rrf_k = rrf_k
        self._semantic_min_score = semantic_min_score
        self._table = None

    def index(self, chunks: list[Chunk]) -> None:
        with tracer.start_as_current_span("file_agent.retriever_index") as span:
            span.set_attribute("file_agent.chunk_count", len(chunks))

            self.clear()
            if not chunks:
                return

            texts = [chunk.text for chunk in chunks]
            embeddings = self._encode(texts)
            records = [
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source_file": str(chunk.metadata.get("source_file", "")),
                    "vector": embeddings[index].tolist(),
                    "metadata_json": json.dumps(
                        chunk.metadata,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
                for index, chunk in enumerate(chunks)
            ]

            self._table = self._connection.create_table(
                self._table_name,
                data=records,
            )
            self._table.create_index(
                "text",
                config=FTS(language=self._fts_language),
            )
            logger.info("Indexed %d chunk(s) into table %r", len(chunks), self._table_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_file: str | None = None,
    ) -> list[SearchResult]:
        with tracer.start_as_current_span("file_agent.retriever_search") as span:
            span.set_attribute("file_agent.query", query)
            span.set_attribute("file_agent.top_k", top_k)
            if source_file is not None:
                span.set_attribute("file_agent.source_file", source_file)
            span.set_attribute("langfuse.observation.type", "retriever")
            span.set_attribute(
                "langfuse.observation.input",
                json.dumps(
                    {"query": query, "top_k": top_k, "source_file": source_file},
                    ensure_ascii=False,
                ),
            )

            if top_k <= 0 or self._table is None:
                span.set_attribute("file_agent.result_count", 0)
                span.set_attribute("langfuse.observation.output", "[]")
                return []

            query = query.strip()
            if not query:
                span.set_attribute("file_agent.result_count", 0)
                span.set_attribute("langfuse.observation.output", "[]")
                return []

            normalized_source = source_file.strip() if source_file is not None else None
            if source_file is not None and not normalized_source:
                span.set_attribute("file_agent.result_count", 0)
                span.set_attribute("langfuse.observation.output", "[]")
                return []

            query_vector = self._encode([query])[0].tolist()
            query_builder = (
                self._table.search(
                    query_type="hybrid",
                    vector_column_name="vector",
                    fts_columns="text",
                )
                .vector(query_vector)
                .text(query)
            )
            if normalized_source is not None:
                escaped_source = normalized_source.replace("'", "''")
                query_builder = query_builder.where(
                    f"source_file = '{escaped_source}'",
                    prefilter=True,
                )
            rows = (
                query_builder.distance_type("cosine")
                .distance_range(upper_bound=1.0 - self._semantic_min_score)
                .rerank(RRFReranker(K=self._rrf_k))
                .limit(top_k)
                .to_list()
            )

            results = [self._to_search_result(row) for row in rows]
            span.set_attribute("file_agent.result_count", len(results))
            span.set_attribute(
                "langfuse.observation.output",
                json.dumps(
                    [{"chunk_id": result.chunk.id, "score": result.score} for result in results],
                    ensure_ascii=False,
                ),
            )
            logger.info("Query %r returned %d result(s)", query, len(results))
            return results

    def clear(self) -> None:
        self._connection.drop_table(self._table_name, ignore_missing=True)
        self._table = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._embedding_model or _load_default_embedding_model(self._embedding_model_name)
        embeddings = model.encode(texts)

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise ValueError("Embedding model returned an unexpected number of embeddings")
        if array.shape[1] == 0:
            raise ValueError("Embedding model returned empty embeddings")

        return array

    @staticmethod
    def _to_search_result(row: dict) -> SearchResult:
        metadata = json.loads(row["metadata_json"])
        return SearchResult(
            chunk=Chunk(
                id=row["chunk_id"],
                text=row["text"],
                metadata=metadata,
            ),
            score=float(row["_relevance_score"]),
        )


def resolve_semantic_model_name(model_name: str | None = None) -> str:
    configured = model_name if model_name is not None else os.getenv("EMBEDDING_MODEL")
    normalized = configured.strip() if configured else ""
    return normalized or DEFAULT_SEMANTIC_MODEL_NAME


@lru_cache(maxsize=2)
def _load_default_embedding_model(model_name: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
