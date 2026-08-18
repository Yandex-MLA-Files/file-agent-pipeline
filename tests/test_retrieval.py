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
