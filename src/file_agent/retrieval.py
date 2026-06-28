from dataclasses import dataclass

from file_agent.chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def search_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    if top_k <= 0:
        return []

    query_words = query.lower().split()
    if not query_words:
        return []

    results: list[SearchResult] = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for word in query_words if word in text)

        if score > 0:
            results.append(SearchResult(chunk=chunk, score=float(score)))

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]
