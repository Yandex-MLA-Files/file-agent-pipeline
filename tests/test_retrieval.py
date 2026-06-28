from file_agent.chunking import Chunk
from file_agent.retrieval import search_chunks


def test_search_chunks_returns_relevant_chunks():
    chunks = [
        Chunk(id="chunk-1", text="Python parses markdown files"),
        Chunk(id="chunk-2", text="Streamlit renders a local demo"),
    ]

    results = search_chunks("python markdown", chunks)

    assert [result.chunk.id for result in results] == ["chunk-1"]
    assert results[0].score == 2.0


def test_search_chunks_does_not_return_irrelevant_chunks():
    chunks = [
        Chunk(id="chunk-1", text="PDF parser extracts text"),
        Chunk(id="chunk-2", text="HTML parser removes scripts"),
    ]

    results = search_chunks("spreadsheet cells", chunks)

    assert results == []


def test_search_chunks_top_k_limits_results():
    chunks = [
        Chunk(id="chunk-1", text="python"),
        Chunk(id="chunk-2", text="python"),
        Chunk(id="chunk-3", text="python"),
    ]

    results = search_chunks("python", chunks, top_k=2)

    assert len(results) == 2
    assert [result.chunk.id for result in results] == ["chunk-1", "chunk-2"]


def test_search_chunks_sorts_results_by_score_descending():
    chunks = [
        Chunk(id="chunk-1", text="python"),
        Chunk(id="chunk-2", text="python markdown html"),
        Chunk(id="chunk-3", text="python markdown"),
    ]

    results = search_chunks("python markdown html", chunks)

    assert [result.chunk.id for result in results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]
    assert [result.score for result in results] == [3.0, 2.0, 1.0]
