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


def test_lancedb_retriever_handles_a_quoted_phrase_query():
    # Regression: FTS defaults to no position data, which makes a phrase
    # query (a quoted substring - LLM-generated search queries sometimes
    # include one) fail outright with "position is not found but required
    # for phrase queries" instead of just running as a normal search.
    chunks = [Chunk(id="a1", text="all authors from the same affiliation")]
    model = FakeEmbeddingModel(
        {
            "all authors from the same affiliation": [1.0, 0.0],
            '"same affiliation"': [1.0, 0.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search('"same affiliation"')

    assert [result.chunk.id for result in results] == ["a1"]


def test_lancedb_retriever_handles_empty_inputs_without_loading_model():
    retriever = LanceDBRetriever(embedding_model=FakeEmbeddingModel({}))

    retriever.index([])

    assert retriever.search("") == []
    assert retriever.search("query", top_k=0) == []


def test_lancedb_retriever_filters_by_source_file():
    chunks = [
        Chunk(id="a1", text="python guide", metadata={"source_file": "a.pdf"}),
        Chunk(id="b1", text="python guide", metadata={"source_file": "b.pdf"}),
    ]
    model = FakeEmbeddingModel({"python guide": [1.0, 0.0], "python": [1.0, 0.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python", source_file="a.pdf")

    assert [result.chunk.id for result in results] == ["a1"]


def test_lancedb_retriever_returns_nothing_for_an_unindexed_source_file():
    chunks = [Chunk(id="a1", text="python guide", metadata={"source_file": "a.pdf"})]
    model = FakeEmbeddingModel({"python guide": [1.0, 0.0], "python": [1.0, 0.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)

    assert retriever.search("python", source_file="missing.pdf") == []


def test_lancedb_retriever_escapes_a_quote_in_source_file():
    # Regression: source_file is interpolated into a SQL where() predicate -
    # a literal apostrophe in a real file name must not break the query or
    # open a SQL-injection path (see search()'s escaping comment).
    chunks = [Chunk(id="a1", text="python guide", metadata={"source_file": "o'brien.pdf"})]
    model = FakeEmbeddingModel({"python guide": [1.0, 0.0], "python": [1.0, 0.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    retriever.index(chunks)
    results = retriever.search("python", source_file="o'brien.pdf")

    assert [result.chunk.id for result in results] == ["a1"]


def test_lancedb_retriever_search_is_safe_under_concurrent_calls():
    # Regression: docbench_pilot.py now answers several questions against one
    # shared retriever concurrently - search() didn't have any locking before,
    # relying on undocumented thread-safety from the embedding model and
    # lancedb's query builder. This doesn't prove the real SentenceTransformer
    # is safe (can't load it in a unit test), but it does prove the lock
    # doesn't deadlock and doesn't scramble results across threads.
    import threading

    chunks = [
        Chunk(id="alpha", text="alpha document"),
        Chunk(id="beta", text="beta document"),
        Chunk(id="gamma", text="gamma document"),
    ]
    model = FakeEmbeddingModel(
        {
            "alpha document": [1.0, 0.0, 0.0],
            "beta document": [0.0, 1.0, 0.0],
            "gamma document": [0.0, 0.0, 1.0],
            "alpha": [1.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
        }
    )
    retriever = LanceDBRetriever(embedding_model=model)
    retriever.index(chunks)

    outcomes: dict[str, list[str]] = {"alpha": [], "beta": [], "gamma": []}
    errors: list[Exception] = []

    def search_repeatedly(query: str) -> None:
        try:
            for _ in range(20):
                results = retriever.search(query, top_k=1)
                outcomes[query].append(results[0].chunk.id if results else "")
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=search_repeatedly, args=(q,)) for q in outcomes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    for query, ids in outcomes.items():
        assert ids == [query] * 20


def test_lancedb_retriever_keeps_embeddings_unnormalized_for_cosine_search():
    model = FakeEmbeddingModel({"document": [3.0, 4.0]})
    retriever = LanceDBRetriever(embedding_model=model)

    embeddings = retriever._encode(["document"])

    assert embeddings.tolist() == [[3.0, 4.0]]
    assert model.calls == [["document"]]
