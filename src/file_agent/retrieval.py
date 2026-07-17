import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from file_agent.chunking import Chunk

DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_MIN_SCORE = 0.25


class SemanticModel(Protocol):
    def encode(self, sentences, normalize_embeddings: bool = True): ...


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def search_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    semantic_model: SemanticModel | None = None,
    use_semantic: bool = True,
    semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
) -> list[SearchResult]:
    if top_k <= 0:
        return []

    query = query.strip()
    if not query or not chunks:
        return []

    rankings: list[list[int]] = []

    bm25_ranking = _rank_chunks_bm25(query, chunks)
    if bm25_ranking:
        rankings.append(bm25_ranking)

    if use_semantic:
        semantic_ranking = _rank_chunks_semantic(
            query=query,
            chunks=chunks,
            semantic_model=semantic_model,
            min_score=semantic_min_score,
        )
        if semantic_ranking:
            rankings.append(semantic_ranking)

    if not rankings:
        return []

    fused_scores = _reciprocal_rank_fusion(rankings)
    return [
        SearchResult(chunk=chunks[chunk_index], score=score)
        for chunk_index, score in fused_scores[:top_k]
    ]


def _rank_chunks_bm25(query: str, chunks: list[Chunk]) -> list[int]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    tokenized_chunks = [_tokenize(chunk.text) for chunk in chunks]
    non_empty_lengths = [len(tokens) for tokens in tokenized_chunks if tokens]
    if not non_empty_lengths:
        return []

    avg_doc_length = sum(non_empty_lengths) / len(non_empty_lengths)
    document_frequencies = _document_frequencies(tokenized_chunks)
    total_documents = len(tokenized_chunks)

    scored_chunks: list[tuple[int, float]] = []
    for chunk_index, document_tokens in enumerate(tokenized_chunks):
        if not document_tokens:
            continue

        score = _bm25_score(
            query_tokens=query_tokens,
            document_tokens=document_tokens,
            document_frequencies=document_frequencies,
            total_documents=total_documents,
            avg_doc_length=avg_doc_length,
        )
        if score > 0:
            scored_chunks.append((chunk_index, score))

    scored_chunks.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_index for chunk_index, _ in scored_chunks]


def _rank_chunks_semantic(
    query: str,
    chunks: list[Chunk],
    semantic_model: SemanticModel | None,
    min_score: float,
) -> list[int]:
    model = semantic_model
    if model is None:
        model = _load_default_semantic_model()
        if model is None:
            return []

    texts = [query] + [chunk.text for chunk in chunks]
    embeddings = _encode_texts(model, texts)
    if embeddings.shape[0] != len(texts):
        raise ValueError("Semantic model returned an unexpected number of embeddings")

    embeddings = _normalize_embeddings(embeddings)
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    similarities = chunk_embeddings @ query_embedding

    scored_chunks = [
        (chunk_index, float(score))
        for chunk_index, score in enumerate(similarities)
        if score >= min_score
    ]
    scored_chunks.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_index for chunk_index, _ in scored_chunks]


def _reciprocal_rank_fusion(
    rankings: list[list[int]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    best_ranks: dict[int, int] = {}

    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] += 1.0 / (rrf_k + rank)
            best_ranks[chunk_index] = min(best_ranks.get(chunk_index, rank), rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_ranks[item[0]], item[0]),
    )


def _bm25_score(
    query_tokens: list[str],
    document_tokens: list[str],
    document_frequencies: dict[str, int],
    total_documents: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    token_counts = Counter(document_tokens)
    document_length = len(document_tokens)
    score = 0.0

    for token in query_tokens:
        term_frequency = token_counts.get(token, 0)
        if term_frequency == 0:
            continue

        doc_frequency = document_frequencies[token]
        idf = math.log(1 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5))
        denominator = term_frequency + k1 * (1 - b + b * document_length / avg_doc_length)
        score += idf * (term_frequency * (k1 + 1)) / denominator

    return score


def _document_frequencies(tokenized_chunks: list[list[str]]) -> dict[str, int]:
    frequencies: dict[str, int] = defaultdict(int)

    for tokens in tokenized_chunks:
        for token in set(tokens):
            frequencies[token] += 1

    return frequencies


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _encode_texts(model: SemanticModel, texts: list[str]) -> np.ndarray:
    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
    except TypeError:
        embeddings = model.encode(texts)

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()

    array = np.asarray(embeddings, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    return array


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0,
    )


@lru_cache(maxsize=1)
def _load_default_semantic_model() -> SemanticModel | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
        return SentenceTransformer(DEFAULT_SEMANTIC_MODEL_NAME)
    except Exception:
        return None
