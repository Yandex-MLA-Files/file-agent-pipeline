import pytest

from file_agent.chunking import Chunk
from file_agent.lancedb_retriever import LanceDBRetriever


class FakeEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, sentences):
        self.calls.append(list(sentences))
        return [self.vectors[sentence] for sentence in sentences]


def test_lancedb_retriever_combines_bm25_and_semantic_results_with_rrf():
    chunks = [
        Chunk(id="lexical", text="python code"),
        Chunk(id="semantic", text="automobile engine"),
        Chunk(id="both", text="python automobile"),
    ]
    model = FakeEmbeddingModel(
        {
            "python code": [0.0, 1.0],
            "automobile engine": [1.0, 0.0],
            "python automobile": [1.0, 0.0],
            "python vehicle": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python vehicle")

    assert results[0].chunk.id == "both"
    assert {result.chunk.id for result in results} == {
        "lexical",
        "semantic",
        "both",
    }
    assert results[0].score > 0


def test_lancedb_retriever_returns_semantic_match_without_keyword_overlap():
    chunks = [
        Chunk(id="vehicle", text="automobile engine"),
        Chunk(id="fruit", text="banana fruit"),
    ]
    model = FakeEmbeddingModel(
        {
            "automobile engine": [1.0, 0.0],
            "banana fruit": [0.0, 1.0],
            "vehicle question": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("vehicle question")

    assert [result.chunk.id for result in results] == ["vehicle"]


def test_lancedb_retriever_uses_russian_stemming_for_bm25():
    chunks = [
        Chunk(id="document", text="Обработка документа", metadata={"source_file": "notes.md"}),
        Chunk(id="other", text="Совсем другой текст"),
    ]
    model = FakeEmbeddingModel(
        {
            "Обработка документа": [1.0, 0.0],
            "Совсем другой текст": [0.0, 1.0],
            "документы": [-1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("документы")

    assert [result.chunk.id for result in results] == ["document"]
    assert results[0].chunk.metadata == {"source_file": "notes.md"}


def test_lancedb_retriever_filters_irrelevant_vector_results():
    chunks = [Chunk(id="parser", text="Python parser")]
    model = FakeEmbeddingModel(
        {
            "Python parser": [1.0, 0.0],
            "spreadsheet cells": [-1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)

    assert retriever.search("spreadsheet cells") == []


def test_lancedb_retriever_honors_top_k_and_clear():
    chunks = [
        Chunk(id="first", text="python first"),
        Chunk(id="second", text="python second"),
        Chunk(id="third", text="python third"),
    ]
    model = FakeEmbeddingModel(
        {
            "python first": [1.0, 0.0],
            "python second": [1.0, 0.0],
            "python third": [1.0, 0.0],
            "python": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    assert len(retriever.search("python", top_k=2)) == 2

    retriever.clear()
    assert retriever.search("python") == []


def test_lancedb_retriever_handles_empty_inputs_without_loading_model():
    retriever = LanceDBRetriever(embedding_model=FakeEmbeddingModel({}))

    retriever.index([])

    assert retriever.search("") == []
    assert retriever.search("query", top_k=0) == []


def test_lancedb_retriever_keeps_embeddings_unnormalized_for_cosine_search():
    model = FakeEmbeddingModel({"document": [3.0, 4.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    embeddings = retriever._encode(["document"])

    assert embeddings.tolist() == [[3.0, 4.0]]
    assert model.calls == [["document"]]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores[text] for _, text in pairs]


def test_optional_reranker_reorders_hybrid_candidates():
    chunks = [
        Chunk(id="a", text="python code"),
        Chunk(id="b", text="python automobile"),
        Chunk(id="c", text="automobile engine"),
    ]
    model = FakeEmbeddingModel(
        {
            "python code": [0.0, 1.0],
            "python automobile": [1.0, 0.0],
            "automobile engine": [1.0, 0.0],
            "python vehicle": [1.0, 0.0],
        }
    )
    reranker = FakeReranker(
        {"python code": 0.9, "python automobile": 0.2, "automobile engine": 0.1}
    )
    retriever = LanceDBRetriever(embedding_model=model, reranker=reranker)

    retriever.index(chunks)
    results = retriever.search("python vehicle", top_k=2)

    assert [result.chunk.id for result in results] == ["a", "b"]
    assert results[0].score == 0.9
    # All hybrid candidates were offered to the reranker, then cut to top_k.
    assert len(reranker.calls[0]) == 3


def test_multi_document_results_include_each_documents_best_hit():
    chunks = [
        Chunk(id="a1", text="python code one", metadata={"source_file": "a.md"}),
        Chunk(id="a2", text="python code two", metadata={"source_file": "a.md"}),
        Chunk(id="a3", text="python code three", metadata={"source_file": "a.md"}),
        Chunk(id="b1", text="python notes", metadata={"source_file": "b.md"}),
    ]
    model = FakeEmbeddingModel(
        {
            "python code one": [1.0, 0.0],
            "python code two": [1.0, 0.0],
            "python code three": [1.0, 0.0],
            "python notes": [0.7, 0.7],
            "python": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python", top_k=2)

    files = {result.chunk.metadata["source_file"] for result in results}
    assert files == {"a.md", "b.md"}


def test_search_can_be_restricted_to_one_document():
    chunks = [
        Chunk(id="a1", text="python code one", metadata={"source_file": "a.md"}),
        Chunk(id="a2", text="python code two", metadata={"source_file": "a.md"}),
        Chunk(id="b1", text="python code three", metadata={"source_file": "b.md"}),
    ]
    model = FakeEmbeddingModel(
        {
            "python code one": [1.0, 0.0],
            "python code two": [1.0, 0.0],
            "python code three": [1.0, 0.0],
            "python code": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python code", top_k=5, source_file="b.md")

    assert [result.chunk.id for result in results] == ["b1"]


def test_source_file_filter_survives_a_quote_in_the_file_name():
    chunks = [
        Chunk(id="quoted", text="python code one", metadata={"source_file": "o'brien.md"}),
        Chunk(id="plain", text="python code two", metadata={"source_file": "b.md"}),
    ]
    model = FakeEmbeddingModel(
        {
            "python code one": [1.0, 0.0],
            "python code two": [1.0, 0.0],
            "python code": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python code", top_k=5, source_file="o'brien.md")

    assert [result.chunk.id for result in results] == ["quoted"]


def test_search_in_a_document_that_was_never_indexed_returns_nothing():
    chunks = [Chunk(id="a1", text="python code one", metadata={"source_file": "a.md"})]
    model = FakeEmbeddingModel({"python code one": [1.0, 0.0], "python code": [1.0, 0.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)

    assert retriever.search("python code", source_file="missing.md") == []


def test_document_diversification_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DIVERSIFY_DOCS", "false")
    chunks = [
        Chunk(id="a1", text="python code one", metadata={"source_file": "a.md"}),
        Chunk(id="a2", text="python code two", metadata={"source_file": "a.md"}),
        Chunk(id="b1", text="python notes", metadata={"source_file": "b.md"}),
    ]
    model = FakeEmbeddingModel(
        {
            "python code one": [1.0, 0.0],
            "python code two": [1.0, 0.0],
            "python notes": [0.0, 1.0],
            "python code": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python code", top_k=2)

    assert {result.chunk.metadata["source_file"] for result in results} == {"a.md"}


class FakeExpander:
    def __init__(self, variants):
        self.variants = variants
        self.calls = []

    def __call__(self, question):
        self.calls.append(question)
        return list(self.variants)


def test_bm25_matches_through_yo_folding_and_lemmas():
    pytest.importorskip("pymorphy3")
    chunks = [
        Chunk(id="yo", text="Ещё один учёт затрат ведётся людьми"),
        Chunk(id="other", text="Совсем посторонний текст"),
    ]
    model = FakeEmbeddingModel(
        {
            "Ещё один учёт затрат ведётся людьми": [0.0, 1.0],
            "Совсем посторонний текст": [1.0, 0.0],
            "еще учет человек": [0.0, -1.0],  # dense side deliberately useless
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("еще учет человек")

    assert [result.chunk.id for result in results] == ["yo"]


def test_quoted_phrase_ranks_the_passages_that_contain_it_first():
    chunks = [
        Chunk(id="words", text="компании увеличили выручку и прибыль"),
        Chunk(id="phrase", text="выручка компании выросла в отчётном году"),
    ]
    model = FakeEmbeddingModel(
        {
            # The dense side prefers the passage that merely shares the words.
            "компании увеличили выручку и прибыль": [1.0, 0.0],
            "выручка компании выросла в отчётном году": [0.8, 0.6],
            '"выручка компании" выросла': [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search('"выручка компании" выросла')

    # Both are still returned (the dense side knows nothing about quotes),
    # but the one that carries the phrase comes first.
    assert [result.chunk.id for result in results] == ["phrase", "words"]


def test_multi_query_variants_are_fused_with_the_original(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_UNIQUE_PASSAGES", "false")
    chunks = [
        Chunk(id="revenue", text="выручка компании за год"),
        Chunk(id="income", text="доходы организации за период"),
        Chunk(id="noise", text="погода в апреле"),
    ]
    model = FakeEmbeddingModel(
        {
            "выручка компании за год": [1.0, 0.0],
            "доходы организации за период": [0.0, 1.0],
            "погода в апреле": [-1.0, 0.0],
            "сколько заработала компания": [1.0, 0.0],
            "доходы организации": [0.0, 1.0],
        }
    )
    expander = FakeExpander(["доходы организации", "сколько заработала компания"])
    retriever = LanceDBRetriever(embedding_model=model, query_expander=expander)

    retriever.index(chunks)
    results = retriever.search("сколько заработала компания", top_k=2)

    assert expander.calls == ["сколько заработала компания"]
    # The variant that says "доходы организации" pulls in a passage the
    # original wording alone would never rank; both relevant chunks are on top.
    assert {result.chunk.id for result in results} == {"revenue", "income"}
    # The duplicate of the original question is not searched twice.
    assert len(model.calls) == 1 + 2  # index call + two distinct query encodes


def test_multi_query_expander_failure_degrades_to_the_single_query():
    chunks = [Chunk(id="a", text="python code")]
    model = FakeEmbeddingModel({"python code": [1.0, 0.0], "python": [1.0, 0.0]})

    def broken(question):
        raise RuntimeError("endpoint down")

    retriever = LanceDBRetriever(embedding_model=model, query_expander=broken)
    retriever.index(chunks)

    with pytest.raises(RuntimeError):
        # A raw callable that raises is the caller's bug; the shipped expander
        # (MultiQueryExpander) swallows its own errors — see test_query_expansion.
        retriever.search("python")


def test_top_k_is_filled_with_distinct_passages(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_UNIQUE_PASSAGES", "true")
    parent = "the whole section text"
    chunks = [
        Chunk(id="s1", text="python code one", metadata={"context": parent}),
        Chunk(id="s2", text="python code two", metadata={"context": parent}),
        Chunk(id="s3", text="python code three", metadata={"context": parent}),
        Chunk(id="other", text="python notes", metadata={"context": "another section"}),
    ]
    model = FakeEmbeddingModel(
        {
            "python code one": [1.0, 0.0],
            "python code two": [1.0, 0.0],
            "python code three": [1.0, 0.0],
            "python notes": [0.9, 0.1],
            "python code": [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python code", top_k=2)

    passages = [result.chunk.metadata["context"] for result in results]
    assert passages == [parent, "another section"]

    monkeypatch.setenv("RETRIEVAL_UNIQUE_PASSAGES", "false")
    results = retriever.search("python code", top_k=2)
    assert [result.chunk.metadata["context"] for result in results] == [parent, parent]


class BatchAwareReranker(FakeReranker):
    def predict(self, pairs, batch_size=None):
        self.batch_sizes = getattr(self, "batch_sizes", []) + [batch_size]
        return super().predict(pairs)


def test_reranker_receives_the_configured_batch_size(monkeypatch):
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "7")
    chunks = [Chunk(id="a", text="python code"), Chunk(id="b", text="python automobile")]
    model = FakeEmbeddingModel(
        {"python code": [1.0, 0.0], "python automobile": [1.0, 0.0], "python": [1.0, 0.0]}
    )
    reranker = BatchAwareReranker({"python code": 0.1, "python automobile": 0.9})
    retriever = LanceDBRetriever(embedding_model=model, reranker=reranker)

    retriever.index(chunks)
    results = retriever.search("python", top_k=2)

    assert reranker.batch_sizes == [7]
    assert [result.chunk.id for result in results] == ["b", "a"]


def test_reranker_blend_can_pull_the_first_stage_order_back(monkeypatch):
    chunks = [Chunk(id="a", text="python code"), Chunk(id="b", text="python automobile")]
    model = FakeEmbeddingModel({"python code": [1.0, 0.0], "python automobile": [0.6, 0.8]})
    # The cross-encoder mildly prefers "b"; the first stage strongly prefers "a".
    reranker = FakeReranker({"python code": 0.49, "python automobile": 0.51})
    retriever = LanceDBRetriever(embedding_model=model, reranker=reranker)
    retriever.index(chunks)

    monkeypatch.setenv("RERANKER_BLEND", "0")
    assert [r.chunk.id for r in retriever.search("python code", top_k=2)] == ["b", "a"]

    monkeypatch.setenv("RERANKER_BLEND", "0.5")
    assert [r.chunk.id for r in retriever.search("python code", top_k=2)] == ["a", "b"]
