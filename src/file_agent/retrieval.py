from dataclasses import dataclass
from typing import Protocol

from file_agent.chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None:
        """Replace the current index contents with the provided chunks."""
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_file: str | None = None,
    ) -> list[SearchResult]:
        """Return the chunks most relevant to the query.

        ``source_file`` limits the answer to one indexed document.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all indexed chunks."""
        raise NotImplementedError
