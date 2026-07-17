from file_agent.chunking import Chunk
from file_agent.retrieval import search_chunks


class FakeSemanticModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, sentences, normalize_embeddings=True):
        self.calls.append((list(sentences), normalize_embeddings))
        return [self.vectors[sentence] for sentence in sentences]


def test_search_chunks_returns_relevant_chunks_with_bm25():
    chunks = [
        Chunk(id="chunk-1", text="Python parses markdown files"),
        Chunk(id="chunk-2", text="Streamlit renders a local demo"),
    ]

    results = search_chunks("python markdown", chunks, use_semantic=False)

    assert [result.chunk.id for result in results] == ["chunk-1"]
    assert results[0].score > 0


def test_search_chunks_does_not_return_irrelevant_chunks_with_bm25():
    chunks = [
        Chunk(id="chunk-1", text="PDF parser extracts text"),
        Chunk(id="chunk-2", text="HTML parser removes scripts"),
    ]

    results = search_chunks("spreadsheet cells", chunks, use_semantic=False)

    assert results == []


def test_search_chunks_top_k_limits_results():
    chunks = [
        Chunk(id="chunk-1", text="python"),
        Chunk(id="chunk-2", text="python"),
        Chunk(id="chunk-3", text="python"),
    ]

    results = search_chunks("python", chunks, top_k=2, use_semantic=False)

    assert len(results) == 2
    assert [result.chunk.id for result in results] == ["chunk-1", "chunk-2"]


def test_search_chunks_sorts_bm25_results_by_score():
    chunks = [
        Chunk(id="chunk-1", text="python"),
        Chunk(id="chunk-2", text="python markdown html"),
        Chunk(id="chunk-3", text="python markdown"),
    ]

    results = search_chunks("python markdown html", chunks, use_semantic=False)

    assert [result.chunk.id for result in results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]


def test_search_chunks_can_return_semantic_matches_without_keyword_overlap():
    chunks = [
        Chunk(id="chunk-1", text="automobile engine"),
        Chunk(id="chunk-2", text="banana fruit"),
    ]
    semantic_model = FakeSemanticModel(
        {
            "vehicle question": [1.0, 0.0],
            "automobile engine": [1.0, 0.0],
            "banana fruit": [0.0, 1.0],
        }
    )

    results = search_chunks(
        "vehicle question",
        chunks,
        semantic_model=semantic_model,
    )

    assert [result.chunk.id for result in results] == ["chunk-1"]
    assert semantic_model.calls == [
        (
            ["vehicle question", "automobile engine", "banana fruit"],
            True,
        )
    ]


def test_search_chunks_uses_rrf_to_combine_bm25_and_semantic_rankings():
    chunks = [
        Chunk(id="lexical", text="python code"),
        Chunk(id="semantic", text="automobile engine"),
        Chunk(id="both", text="python automobile"),
    ]
    semantic_model = FakeSemanticModel(
        {
            "python vehicle": [1.0, 0.0],
            "python code": [0.0, 1.0],
            "automobile engine": [1.0, 0.0],
            "python automobile": [1.0, 0.0],
        }
    )

    results = search_chunks(
        "python vehicle",
        chunks,
        semantic_model=semantic_model,
    )

    assert results[0].chunk.id == "both"
    assert {result.chunk.id for result in results} == {
        "lexical",
        "semantic",
        "both",
    }
