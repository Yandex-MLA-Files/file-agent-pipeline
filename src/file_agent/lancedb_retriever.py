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

# BGE-M3: multilingual (strong on Russian), 8192-token window, dense retrieval
# quality far above the 128-token MiniLM it replaces. The previous encoder
# stays available with EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
DEFAULT_SEMANTIC_MODEL_NAME = "BAAI/bge-m3"
LEGACY_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FTS_LANGUAGE = "Russian"
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_MIN_SCORE = 0.25
DEFAULT_TABLE_NAME = "chunks"
# Optional second stage: a cross-encoder re-scores the top hybrid candidates
# with the query and the chunk text side by side. ``RERANKER_MODEL`` (e.g.
# ``BAAI/bge-reranker-v2-m3``) turns it on; ``RERANKER_CANDIDATES`` is how many
# hybrid hits are re-scored (default 4x top_k, at least 20).
DEFAULT_RERANKER_CANDIDATES_FACTOR = 4
DEFAULT_RERANKER_MIN_CANDIDATES = 20
# With several indexed documents, one document can monopolise the top-k for a
# question that spans two files ("compare A and B"). Diversification keeps the
# best hit of every document that appears among the candidates before filling
# the remaining slots by score. ``RETRIEVAL_DIVERSIFY_DOCS=false`` disables it.
DEFAULT_DIVERSIFY_DOCS = True
DEFAULT_DIVERSIFY_CANDIDATES = 20


class EmbeddingModel(Protocol):
    def encode(self, sentences): ...


class Reranker(Protocol):
    def predict(self, pairs): ...


class LanceDBRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        uri: str = "memory://",
        table_name: str = DEFAULT_TABLE_NAME,
        fts_language: str = DEFAULT_FTS_LANGUAGE,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
        reranker: Reranker | None = None,
    ) -> None:
        if not -1.0 <= semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between -1 and 1")

        self._embedding_model = embedding_model
        self._reranker = reranker
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
                    # Own column rather than a field of ``metadata_json``: it is
                    # the only metadata a query filters on, and LanceDB cannot
                    # filter inside a JSON string.
                    "source_file": str((chunk.metadata or {}).get("source_file") or ""),
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
        """Retrieve the ``top_k`` passages best matching ``query``.

        ``source_file`` restricts the search to one indexed document, which is
        what a per-document tool ("what does this file say about X?") needs:
        without it the same question over a corpus of twenty files answers from
        whichever file happens to score highest.
        """
        with tracer.start_as_current_span("file_agent.retriever_search") as span:
            span.set_attribute("file_agent.query", query)
            span.set_attribute("file_agent.top_k", top_k)
            if source_file:
                span.set_attribute("file_agent.source_file", source_file)

            if top_k <= 0 or self._table is None:
                span.set_attribute("file_agent.result_count", 0)
                return []

            query = query.strip()
            source_file = source_file.strip() if source_file is not None else None
            if not query:
                span.set_attribute("file_agent.result_count", 0)
                return []

            reranker = self._reranker or _load_default_reranker()
            # One document cannot be diversified against itself.
            diversify = not source_file and _bool_env(
                "RETRIEVAL_DIVERSIFY_DOCS", DEFAULT_DIVERSIFY_DOCS
            )
            candidate_count = top_k
            if reranker is not None:
                candidate_count = max(
                    top_k * DEFAULT_RERANKER_CANDIDATES_FACTOR,
                    DEFAULT_RERANKER_MIN_CANDIDATES,
                    _int_env("RERANKER_CANDIDATES", 0),
                )
            if diversify:
                candidate_count = max(candidate_count, DEFAULT_DIVERSIFY_CANDIDATES, top_k)

            query_vector = self._encode([query])[0].tolist()
            builder = (
                self._table.search(
                    query_type="hybrid",
                    vector_column_name="vector",
                    fts_columns="text",
                )
                .vector(query_vector)
                .text(query)
                .distance_type("cosine")
                .distance_range(upper_bound=1.0 - self._semantic_min_score)
                .rerank(RRFReranker(K=self._rrf_k))
            )
            if source_file:
                # Single quotes are the SQL string delimiter, so a file name
                # containing one would end the literal (and a crafted name could
                # append a predicate); doubling escapes it.
                escaped = source_file.replace("'", "''")
                builder = builder.where(f"source_file = '{escaped}'", prefilter=True)
            rows = builder.limit(candidate_count).to_list()

            results = [self._to_search_result(row) for row in rows]
            if reranker is not None and len(results) > 1:
                results = self._rerank(reranker, query, results)
            if diversify:
                results = self._diversify_by_document(results, top_k)
            results = results[:top_k]
            span.set_attribute("file_agent.result_count", len(results))
            span.set_attribute("file_agent.reranked", reranker is not None)
            logger.info("Query %r returned %d result(s)", query, len(results))
            return results

    @staticmethod
    def _diversify_by_document(results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Guarantee the best hit of each document a slot, then fill by score.

        Results are already sorted by relevance. When candidates come from
        several files, the first pass takes the top hit of each file in score
        order (bounded by ``top_k``); the second pass appends the remaining
        results in their original order. With a single document this is the
        identity.
        """
        seen_docs: set[str] = set()
        head: list[SearchResult] = []
        for result in results:
            doc = str(result.chunk.metadata.get("source_file") or "")
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            head.append(result)
            if len(head) >= top_k:
                break
        if len(seen_docs) <= 1:
            return results
        chosen = {id(result) for result in head}
        tail = [result for result in results if id(result) not in chosen]
        return head + tail

    @staticmethod
    def _rerank(reranker: Reranker, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-score candidates with a cross-encoder; the score becomes the relevance."""
        pairs = [(query, result.chunk.text) for result in results]
        scores = reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        rescored = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores, strict=True)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    def clear(self) -> None:
        self._connection.drop_table(self._table_name, ignore_missing=True)
        self._table = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._embedding_model or _load_default_embedding_model()
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


def resolve_embedding_model_name() -> str:
    return (os.getenv("EMBEDDING_MODEL") or DEFAULT_SEMANTIC_MODEL_NAME).strip()


@lru_cache(maxsize=2)
def _load_embedding_model(name: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    try:
        import torch

        if torch.cuda.is_available():
            # Half precision halves GPU memory and doubles throughput with no
            # measurable retrieval difference; the GPU is shared with the LLM.
            model.half()
    except Exception:  # pragma: no cover - torch missing or CPU-only build
        pass
    logger.info("Loaded embedding model %s", name)
    return model


def _load_default_embedding_model() -> EmbeddingModel:
    return _load_embedding_model(resolve_embedding_model_name())


def resolve_reranker_model_name() -> str | None:
    name = (os.getenv("RERANKER_MODEL") or "").strip()
    return name or None


@lru_cache(maxsize=2)
def _load_reranker(name: str) -> Reranker:
    from sentence_transformers import CrossEncoder

    kwargs = {}
    try:
        import torch

        if torch.cuda.is_available():
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    except Exception:  # pragma: no cover
        pass
    model = CrossEncoder(name, max_length=1024, **kwargs)
    logger.info("Loaded reranker %s", name)
    return model


def _load_default_reranker() -> Reranker | None:
    name = resolve_reranker_model_name()
    return _load_reranker(name) if name else None


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
