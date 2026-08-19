"""Every piece of document text the agent has seen, under a stable id.

Tools label what they show the model as ``[P1]``, ``[P2]``, ... and the Final
Answer names the ids it relied on. Those passages become the answer's sources:
what the UI lists under the answer and what the evaluation judges the answer
against. Keeping the registry per run (rather than per tool) gives one id
space across tools and steps, so "P3" means the same thing on step 2 and in
the final citation line.
"""

from dataclasses import dataclass, field
from typing import Any

from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult

# Block metadata that ties a passage back to the evaluation dataset; tool-made
# passages copy it from the blocks they were cut from so they export like
# retrieved chunks do.
_INHERITED_METADATA = ("dataset_doc_id", "dataset_record_id")


@dataclass
class Passage:
    """One labelled piece of text shown to the model."""

    id: str
    text: str
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)
    result: SearchResult | None = None

    def header(self, extra: str | None = None) -> str:
        """``[P3 | file=report.pdf | section=Results | pages=4, 5]``."""
        parts = [self.id, f"file={self.source_file or 'unknown'}"]
        if extra:
            parts.append(extra)
        parts.extend(describe_location(self.metadata))
        return f"[{' | '.join(parts)}]"

    def render(self, extra: str | None = None, max_chars: int | None = None) -> str:
        text = self.text
        if max_chars is not None and len(text) > max_chars:
            text = (
                text[:max_chars].rstrip()
                + f"\n[... {self.id} truncated; ask for it with read tools]"
            )
        return f"{self.header(extra)}\n{text}"

    def as_search_result(self) -> SearchResult:
        if self.result is not None:
            return self.result
        return SearchResult(
            chunk=Chunk(id=f"agent-{self.id}", text=self.text, metadata=dict(self.metadata)),
            score=0.0,
        )


def describe_location(metadata: dict[str, Any]) -> list[str]:
    """Human-readable coordinates of a passage inside its document."""
    parts: list[str] = []
    if metadata.get("section"):
        parts.append(f"section={metadata['section']}")
    if metadata.get("sheet_name"):
        parts.append(f"sheet={metadata['sheet_name']}")
    pages = metadata.get("page_numbers") or (
        [metadata["page_number"]] if metadata.get("page_number") else []
    )
    if pages:
        parts.append(f"pages={', '.join(str(page) for page in pages)}")
    elif metadata.get("slide_number"):
        parts.append(f"slide={metadata['slide_number']}")
    return parts


class PassageRegistry:
    """Assigns ids to passages and remembers them for citation resolution."""

    def __init__(self) -> None:
        self._passages: list[Passage] = []
        self._by_key: dict[tuple[str, str], Passage] = {}

    def __len__(self) -> int:
        return len(self._passages)

    def all(self) -> list[Passage]:
        return list(self._passages)

    def get(self, passage_id: str) -> Passage | None:
        wanted = passage_id.strip().upper()
        for passage in self._passages:
            if passage.id == wanted:
                return passage
        return None

    def add_result(self, result: SearchResult, tool: str | None = None) -> Passage:
        """Register a retrieval hit; the parent passage is what the model reads."""
        metadata = dict(result.chunk.metadata)
        text = metadata.pop("context", None) or result.chunk.text
        source_file = str(metadata.get("source_file") or "")
        key = (source_file, text)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        if tool:
            metadata.setdefault("tool", tool)
        passage = Passage(
            id=self._next_id(),
            text=text,
            source_file=source_file,
            metadata=metadata,
            result=result,
        )
        self._register(key, passage)
        return passage

    def add_text(
        self,
        text: str,
        source_file: str,
        metadata: dict[str, Any] | None = None,
        tool: str | None = None,
        inherit_from: dict[str, Any] | None = None,
    ) -> Passage:
        """Register text a tool assembled itself (a section, a grep hit, a computed table)."""
        key = (source_file, text)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        combined: dict[str, Any] = {"source_file": source_file}
        if inherit_from:
            for name in _INHERITED_METADATA:
                if name in inherit_from:
                    combined[name] = inherit_from[name]
        if metadata:
            combined.update(metadata)
        if tool:
            combined["tool"] = tool
        passage = Passage(id=self._next_id(), text=text, source_file=source_file, metadata=combined)
        self._register(key, passage)
        return passage

    def _next_id(self) -> str:
        return f"P{len(self._passages) + 1}"

    def _register(self, key: tuple[str, str], passage: Passage) -> None:
        self._passages.append(passage)
        self._by_key[key] = passage
