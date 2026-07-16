import json
from functools import lru_cache
from typing import Protocol

import lancedb
import numpy as np
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult

DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FTS_LANGUAGE = "Russian"
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_MIN_SCORE = 0.25
DEFAULT_TABLE_NAME = "chunks"


class EmbeddingModel(Protocol):
    def encode(self, sentences, normalize_embeddings: bool = True): ...


class LanceDBRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        uri: str = "memory://",
        table_name: str = DEFAULT_TABLE_NAME,
        fts_language: str = DEFAULT_FTS_LANGUAGE,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
    ) -> None:
        if not -1.0 <= semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between -1 and 1")

        self._embedding_model = embedding_model
        self._connection = lancedb.connect(uri)
        self._table_name = table_name
        self._fts_language = fts_language
        self._rrf_k = rrf_k
        self._semantic_min_score = semantic_min_score
        self._table = None

    def index(self, chunks: list[Chunk]) -> None:
        self.clear()
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self._encode(texts)
        records = [
            {
                "chunk_id": chunk.id,
                "text": chunk.text,
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

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0 or self._table is None:
            return []

        query = query.strip()
        if not query:
            return []

        query_vector = self._encode([query])[0].tolist()
        rows = (
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
            .limit(top_k)
            .to_list()
        )

        return [self._to_search_result(row) for row in rows]

    def clear(self) -> None:
        self._connection.drop_table(self._table_name, ignore_missing=True)
        self._table = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._embedding_model or _load_default_embedding_model()
        try:
            embeddings = model.encode(texts, normalize_embeddings=True)
        except TypeError:
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

        return _normalize_embeddings(array)

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


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0,
    )


@lru_cache(maxsize=1)
def _load_default_embedding_model() -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_SEMANTIC_MODEL_NAME)
