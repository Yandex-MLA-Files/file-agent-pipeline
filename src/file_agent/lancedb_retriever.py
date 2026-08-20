import json
import logging
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import lancedb
import numpy as np
from lancedb.index import FTS
from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery, Occur, PhraseQuery
from lancedb.rerankers import RRFReranker

from file_agent.chunking import Chunk
from file_agent.query_expansion import (
    MultiQuerySettings,
    QueryExpander,
    resolve_query_expander,
)
from file_agent.retrieval import SearchResult
from file_agent.telemetry import tracer
from file_agent.text_normalization import lemmatization_enabled, normalize_for_fts

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
# hybrid hits are re-scored (default 6x top_k, at least 30 — a cross-encoder can
# only promote what the first stage hands it, and the document that answers a
# question worded differently from the way it is written often sits at rank
# 20-40 of the hybrid list). ``RERANKER_BATCH_SIZE`` and ``RERANKER_MAX_LENGTH``
# bound the cost of one pass; ``RERANKER_BLEND`` (0..1) mixes the hybrid RRF
# rank back into the final order (0 = the cross-encoder decides alone).
DEFAULT_RERANKER_CANDIDATES_FACTOR = 6
DEFAULT_RERANKER_MIN_CANDIDATES = 30
DEFAULT_RERANKER_BATCH_SIZE = 32
DEFAULT_RERANKER_MAX_LENGTH = 1024
DEFAULT_RERANKER_BLEND = 0.0
# Multi-query fusion: each formulation of the question runs the same hybrid
# search; the lists are fused with reciprocal-rank fusion. The original
# question is weighted 1.0, generated variants ``MULTI_QUERY_WEIGHT``.
DEFAULT_MULTI_QUERY_WEIGHT = 1.0
# The prompt shows the parent passage of a chunk, so two chunks of the same
# section would be read as one passage — the second one is a wasted slot.
# ``RETRIEVAL_UNIQUE_PASSAGES`` fills top-k with distinct passages instead.
DEFAULT_UNIQUE_PASSAGES = True
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
        query_expander: QueryExpander | None = None,
    ) -> None:
        if not -1.0 <= semantic_min_score <= 1.0:
            raise ValueError("semantic_min_score must be between -1 and 1")

        self._embedding_model = embedding_model
        self._reranker = reranker
        # ``None`` means "whatever the environment says" (``MULTI_QUERY``);
        # inject a callable to pin it, e.g. in tests or a benchmark.
        self._query_expander = query_expander
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
                    # Lexical form for BM25: ё folded, lemmatised when
                    # pymorphy3 is available. The query goes through the same
                    # function, see ``_lexical_query``.
                    "fts_text": normalize_for_fts(chunk.text),
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
            # Two lexical views of every chunk: the raw text (stemmed by the
            # index) and the normalised one. A query matches through either,
            # so a lemma the analyser gets wrong still meets its stem.
            self._table.create_index("text", config=FTS(language=self._fts_language))
            self._table.create_index(
                "fts_text",
                config=FTS(language=self._fts_language, with_position=True),
            )
            logger.info("Indexed %d chunk(s) into table %r", len(chunks), self._table_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_file: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve the ``top_k`` passages best matching ``query``.

        The stages, each optional and each measured on its own:

        1. *multi-query* — the question plus its generated formulations, one
           hybrid (BM25 + dense, RRF) search each, fused by reciprocal rank;
        2. *rerank* — a cross-encoder re-scores the fused candidates against the
           original question;
        3. *diversify* — the best hit of every document keeps a slot;
        4. *unique passages* — top-k is filled with distinct parent passages.

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
            unique_passages = _bool_env("RETRIEVAL_UNIQUE_PASSAGES", DEFAULT_UNIQUE_PASSAGES)
            candidate_count = top_k
            if reranker is not None:
                candidate_count = max(
                    top_k * DEFAULT_RERANKER_CANDIDATES_FACTOR,
                    DEFAULT_RERANKER_MIN_CANDIDATES,
                    _int_env("RERANKER_CANDIDATES", 0),
                )
            if diversify:
                candidate_count = max(candidate_count, DEFAULT_DIVERSIFY_CANDIDATES, top_k)
            if unique_passages:
                # Several chunks per passage is the normal case; ask for enough
                # candidates that top-k distinct passages exist among them.
                candidate_count = max(candidate_count, top_k * 3)

            queries = self._formulations(query)
            span.set_attribute("file_agent.query_count", len(queries))
            ranked_lists = [
                self._hybrid_candidates(text, candidate_count, source_file) for text in queries
            ]
            if len(ranked_lists) == 1:
                results = ranked_lists[0]
            else:
                variant_weight = _float_env("MULTI_QUERY_WEIGHT", DEFAULT_MULTI_QUERY_WEIGHT)
                weights = [1.0] + [variant_weight] * (len(ranked_lists) - 1)
                results = _reciprocal_rank_fusion(ranked_lists, weights, self._rrf_k)
                results = results[: max(candidate_count, top_k)]
            results = _prefer_quoted_phrases(query, results)

            if reranker is not None and len(results) > 1:
                results = self._rerank(reranker, query, results)
            if diversify:
                results = self._diversify_by_document(results, top_k)
            if unique_passages:
                results = self._unique_passages(results, top_k)
            results = results[:top_k]
            span.set_attribute("file_agent.result_count", len(results))
            span.set_attribute("file_agent.reranked", reranker is not None)
            logger.info("Query %r returned %d result(s)", query, len(results))
            return results

    def _formulations(self, query: str) -> list[str]:
        """The original query first, then whatever the expander adds."""
        expander = self._query_expander
        if expander is None:
            expander = resolve_query_expander()
        if expander is None:
            return [query]
        variants = expander(query)
        seen = {query.strip().lower()}
        formulations = [query]
        for variant in variants:
            key = variant.strip().lower()
            if variant.strip() and key not in seen:
                seen.add(key)
                formulations.append(variant.strip())
        return formulations

    def _hybrid_candidates(
        self, query: str, limit: int, source_file: str | None
    ) -> list[SearchResult]:
        """One hybrid search: dense + lexical, fused by LanceDB's RRF."""
        query_vector = self._encode([query])[0].tolist()
        builder = (
            self._table.search(
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns=["text", "fts_text"],
            )
            .vector(query_vector)
            .text(_lexical_query(query))
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
        rows = builder.limit(limit).to_list()
        return [self._to_search_result(row) for row in rows]

    @staticmethod
    def _unique_passages(results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Keep the first result of every distinct passage until top-k are found.

        The passage is what the prompt shows — the parent ``context`` when the
        chunk has one, the chunk text otherwise — so this is exactly the
        collapsing the prompt builder does, moved before the cut so that the
        cut leaves ``top_k`` passages rather than ``top_k`` chunks.
        """
        seen: set[str] = set()
        head: list[SearchResult] = []
        for result in results:
            passage = result.chunk.metadata.get("context") or result.chunk.text
            if passage in seen:
                continue
            seen.add(passage)
            head.append(result)
            if len(head) >= top_k:
                break
        return head

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
        """Re-score candidates with a cross-encoder; the score becomes the relevance.

        With ``RERANKER_BLEND`` > 0 the first-stage order is mixed back in as a
        reciprocal-rank term, which guards against a cross-encoder that is
        confidently wrong on one pair; at 0 (default) it decides alone.
        """
        pairs = [(query, result.chunk.text) for result in results]
        try:
            scores = reranker.predict(
                pairs, batch_size=_int_env("RERANKER_BATCH_SIZE", DEFAULT_RERANKER_BATCH_SIZE)
            )
        except TypeError:  # a reranker that does not take batch_size
            scores = reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = [float(score) for score in scores]
        blend = _float_env("RERANKER_BLEND", DEFAULT_RERANKER_BLEND)
        if blend > 0 and len(scores) > 1:
            low, high = min(scores), max(scores)
            span = (high - low) or 1.0
            first_stage = [1.0 / (DEFAULT_RRF_K + rank) for rank in range(1, len(scores) + 1)]
            fs_low, fs_high = min(first_stage), max(first_stage)
            fs_span = (fs_high - fs_low) or 1.0
            scores = [
                (1 - blend) * (score - low) / span + blend * (stage - fs_low) / fs_span
                for score, stage in zip(scores, first_stage, strict=True)
            ]
        rescored = [
            SearchResult(chunk=result.chunk, score=score)
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


def retrieval_settings_fingerprint() -> dict[str, object]:
    """The retrieval settings that decide which passages a question sees.

    Recorded in run manifests and checkpoint fingerprints, so a row generated
    under one configuration is never resumed under another.
    """
    multi_query = MultiQuerySettings.from_env()
    return {
        "multi_query": multi_query.enabled,
        "multi_query_mode": multi_query.mode if multi_query.enabled else None,
        "multi_query_count": multi_query.count if multi_query.enabled else None,
        "multi_query_weight": _float_env("MULTI_QUERY_WEIGHT", DEFAULT_MULTI_QUERY_WEIGHT),
        "bm25_lemmatize": lemmatization_enabled(),
        "reranker_candidates": _int_env("RERANKER_CANDIDATES", 0) or None,
        "reranker_max_length": _int_env("RERANKER_MAX_LENGTH", DEFAULT_RERANKER_MAX_LENGTH),
        "reranker_blend": _float_env("RERANKER_BLEND", DEFAULT_RERANKER_BLEND),
        "unique_passages": _bool_env("RETRIEVAL_UNIQUE_PASSAGES", DEFAULT_UNIQUE_PASSAGES),
        "diversify_docs": _bool_env("RETRIEVAL_DIVERSIFY_DOCS", DEFAULT_DIVERSIFY_DOCS),
    }


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
    max_length = _int_env("RERANKER_MAX_LENGTH", DEFAULT_RERANKER_MAX_LENGTH)
    model = CrossEncoder(name, max_length=max_length, **kwargs)
    logger.info("Loaded reranker %s", name)
    return model


def _load_default_reranker() -> Reranker | None:
    name = resolve_reranker_model_name()
    return _load_reranker(name) if name else None


_QUOTED_PHRASE = re.compile(r'"([^"]{2,})"|«([^»]{2,})»')


def _lexical_query(query: str) -> FullTextQuery:
    """The BM25 side of a hybrid search.

    The raw words go against the stemmed ``text`` index and their normalised
    form (ё folded, lemmatised) against ``fts_text``; a chunk scores through
    whichever view matches. A phrase the user put in quotes becomes a
    positional phrase query that every hit MUST contain.
    """
    clauses: list[tuple[Occur, FullTextQuery]] = []
    for match in _QUOTED_PHRASE.finditer(query):
        phrase = normalize_for_fts(match.group(1) or match.group(2))
        if len(phrase.split()) >= 2:
            clauses.append((Occur.MUST, PhraseQuery(phrase, "fts_text")))
    unquoted = _QUOTED_PHRASE.sub(" ", query).strip() or query
    clauses.append((Occur.SHOULD, MatchQuery(unquoted, "text")))
    normalized = normalize_for_fts(unquoted)
    if normalized:
        clauses.append((Occur.SHOULD, MatchQuery(normalized, "fts_text")))
    return BooleanQuery(clauses)


def _prefer_quoted_phrases(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Move hits that contain every quoted phrase of the query ahead of the rest.

    The lexical side already requires the phrase, but the dense side of a
    hybrid search knows nothing about quotes; ordering rather than filtering
    keeps recall when the phrase is spelled slightly differently in the text
    (OCR, a hyphen, a case ending) while honouring the user's emphasis.
    """
    phrases = [
        normalize_for_fts(match.group(1) or match.group(2))
        for match in _QUOTED_PHRASE.finditer(query)
    ]
    phrases = [phrase for phrase in phrases if len(phrase.split()) >= 2]
    if not phrases or len(results) < 2:
        return results
    with_phrase: list[SearchResult] = []
    without: list[SearchResult] = []
    for result in results:
        haystack = f" {normalize_for_fts(result.chunk.text)} "
        if all(f" {phrase} " in haystack for phrase in phrases):
            with_phrase.append(result)
        else:
            without.append(result)
    return with_phrase + without


def _reciprocal_rank_fusion(
    ranked_lists: Sequence[list[SearchResult]], weights: Sequence[float], k: int
) -> list[SearchResult]:
    """Fuse several rankings of the same index: score = sum of w / (k + rank)."""
    fused: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for results, weight in zip(ranked_lists, weights, strict=True):
        for rank, result in enumerate(results, start=1):
            key = result.chunk.id
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
            first_seen.setdefault(key, result)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [SearchResult(chunk=first_seen[key].chunk, score=score) for key, score in ordered]


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
