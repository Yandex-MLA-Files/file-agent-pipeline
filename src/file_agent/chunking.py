import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from file_agent.document import Block, BlockType, Document
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 100
# Upper bound for an auto-detected token budget: most sentence-transformers
# encoders top out at 512 positions, and tokenizers often report a sentinel
# value (e.g. 1e30) as ``model_max_length``.
MAX_AUTO_TOKENS = 512

_SEPARATOR = "\n\n"

# Per-block Docling internals that are meaningless once blocks are packed together.
_SKIP_BLOCK_METADATA = frozenset({"docling_label", "hierarchy_level"})

# Upper bound for the parent passage stored in chunk metadata (small-to-big
# retrieval): small chunks give precise embeddings, but the LLM answers from the
# surrounding section, so each chunk carries its parent text up to this size.
PARENT_CONTEXT_MAX_CHARS = 4000

# A table header is repeated on every piece only while it stays this small a
# share of the budget; a huge header would crowd out the actual data rows.
HEADER_REPEAT_MAX_RATIO = 0.25

# Sentence boundary: end punctuation (Latin or Cyrillic text) followed by space,
# or an explicit line break. Used to avoid cutting a chunk mid-sentence.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


class Tokenizer(Protocol):
    """Minimal interface satisfied by HuggingFace tokenizers."""

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk for export (CSV/JSON) or storage in a vector DB."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class _Section:
    """A heading and the blocks that belong to it (its reading-order body)."""

    heading: str | None
    blocks: list[Block]

    @property
    def text(self) -> str:
        return _SEPARATOR.join(b.text for b in self.blocks if b.text)


class _Budget:
    """Chunk size budget, measured in tokens when possible, characters otherwise.

    Character counts are only a proxy for what an embedding model actually sees:
    it truncates at a fixed number of *tokens*, and token density differs per
    language (Cyrillic text costs more tokens per character than English). When
    a tokenizer is supplied we therefore budget in real tokens, which keeps
    chunks inside the encoder's window in every language.
    """

    def __init__(self, limit: int, minimum: int, overlap: int, tokenizer: Tokenizer | None):
        self.limit = limit
        self.minimum = minimum
        self.overlap = overlap
        self._tokenizer = tokenizer

    @property
    def unit(self) -> str:
        return "tokens" if self._tokenizer is not None else "chars"

    def size(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text)
        try:
            # verbose=False silences the tokenizer's "sequence longer than the
            # model maximum" notice: measuring long text is exactly the point.
            return len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))
        except TypeError:  # tokenizers that do not accept those keywords
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(self._tokenizer.encode(text))
        except Exception:  # pragma: no cover - never fail chunking on tokenizer issues
            return len(text)


def _build_budget(
    max_chars: int,
    overlap: int,
    min_chars: int | None,
    max_tokens: int | None,
    tokenizer: Tokenizer | None,
) -> _Budget:
    if tokenizer is None:
        limit = max_chars
        minimum = max(1, max_chars // 3) if min_chars is None else min(min_chars, max_chars)
        return _Budget(limit=limit, minimum=minimum, overlap=overlap, tokenizer=None)

    limit = max_tokens or _tokenizer_limit(tokenizer)
    # Keep the caller's overlap *ratio* when switching units, so the defaults
    # (100 of 1000 chars) stay a sensible 10% in token space too.
    scaled_overlap = round(limit * overlap / max_chars) if max_chars else 0
    return _Budget(
        limit=limit,
        minimum=max(1, limit // 3),
        overlap=max(0, min(scaled_overlap, limit - 1)),
        tokenizer=tokenizer,
    )


def _tokenizer_limit(tokenizer: Tokenizer) -> int:
    raw = getattr(tokenizer, "model_max_length", None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MAX_AUTO_TOKENS
    if value <= 0 or value > MAX_AUTO_TOKENS:
        return MAX_AUTO_TOKENS
    return value


@lru_cache(maxsize=4)
def get_embedding_tokenizer(model_name: str | None = None) -> Tokenizer | None:
    """Return the tokenizer of the retrieval embedding model, or None.

    Chunk sizes should match what the encoder can actually embed; anything past
    its window is silently dropped at index time. Loading is lazy and failures
    (offline environment, missing extra) degrade to character budgeting.
    """
    name = model_name or os.getenv("EMBEDDING_MODEL")
    if not name:
        from file_agent.lancedb_retriever import DEFAULT_SEMANTIC_MODEL_NAME

        name = DEFAULT_SEMANTIC_MODEL_NAME
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception:
        logger.info("Embedding tokenizer unavailable; chunking by characters.", exc_info=True)
        return None

    # sentence-transformers caps sequences below the tokenizer's own maximum;
    # honour that limit so chunks are never truncated at index time.
    limit = _sentence_transformers_limit(name)
    if limit:
        tokenizer.model_max_length = limit
    return tokenizer


def _sentence_transformers_limit(model_name: str) -> int | None:
    try:
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "sentence_bert_config.json")
        with open(path, encoding="utf-8") as handle:
            return int(json.load(handle).get("max_seq_length")) or None
    except Exception:
        return None


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int | None = None,
    max_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Split a document into retrieval-sized, section-coherent chunks.

    The chunker follows three principles used by production RAG stacks:

    1. **Structure first.** Blocks are grouped into sections (a heading plus its
       body), so a heading always opens a chunk and never dangles at the end of
       an unrelated one. Whole sections are then packed together up to the size
       budget, which keeps small slides/paragraphs from becoming useless
       single-sentence chunks while never mixing a section into a chunk that is
       already large enough to stand on its own.
    2. **Budget in the encoder's unit.** When ``tokenizer`` is provided, chunk
       size is measured in tokens (default: the encoder's own window, see
       :func:`get_embedding_tokenizer`) instead of characters, so nothing is
       silently truncated at index time and Russian and English text are treated
       consistently. Without a tokenizer the budget falls back to characters.
    3. **Split on natural boundaries.** Oversized blocks are divided at sentence
       boundaries (then words, then characters as a last resort) rather than
       mid-word, tables are split by rows repeating the header, and continuation
       chunks keep their section heading as a breadcrumb.

    Each chunk records the section it starts under, all sections it covers, page
    numbers, block ids and any VLM description for filtering and tracing.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    with tracer.start_as_current_span("file_agent.chunk_document") as span:
        span.set_attribute("file_agent.block_count", len(document.blocks))
        span.set_attribute("file_agent.max_chars", max_chars)
        span.set_attribute("file_agent.overlap", overlap)

        budget = _build_budget(max_chars, overlap, min_chars, max_tokens, tokenizer)
        chunker = _Chunker(document, budget)
        for section in _group_sections(document.blocks):
            chunker.add_section(section)
        chunks = chunker.finish()

        span.set_attribute("file_agent.chunk_count", len(chunks))
        logger.info("Chunked %d block(s) into %d chunk(s)", len(document.blocks), len(chunks))
        return chunks


def _group_sections(blocks: list[Block]) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(heading=None, blocks=[])
    for block in blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            if current.blocks:
                sections.append(current)
            current = _Section(heading=block.text.strip(), blocks=[block])
        else:
            current.blocks.append(block)
    if current.blocks:
        sections.append(current)
    return sections


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text)]
    return [part for part in parts if part]


class _Chunker:
    def __init__(self, document: Document, budget: _Budget) -> None:
        self._document = document
        self._budget = budget
        # Packed pieces are joined by separators, which cost budget too.
        self._separator_size = budget.size(_SEPARATOR)
        self._line_size = budget.size("\n")
        self._chunks: list[Chunk] = []
        self._index = 1
        self._buffer: list[_Section] = []
        self._buffer_size = 0

    def add_section(self, section: _Section) -> None:
        text = section.text
        if not text:
            return

        size = self._budget.size(text)
        if size > self._budget.limit:
            self._flush()
            self._pack_blocks(section.blocks, section.heading)
            return

        addition = size + (self._separator_size if self._buffer else 0)
        would_exceed = self._buffer_size + addition > self._budget.limit
        if self._buffer and (self._buffer_size >= self._budget.minimum or would_exceed):
            self._flush()
            addition = size

        self._buffer.append(section)
        self._buffer_size += addition

    def finish(self) -> list[Chunk]:
        self._flush()
        return self._chunks

    # -- internals ----------------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer:
            return
        blocks = [block for section in self._buffer for block in section.blocks]
        headings = [section.heading for section in self._buffer if section.heading]
        self._emit(blocks, section=headings[0] if headings else None, sections=headings)
        self._buffer = []
        self._buffer_size = 0

    def _reserved_limit(self, heading: str | None) -> int:
        """Budget available for content once the breadcrumb heading is added."""
        if not heading:
            return self._budget.limit
        reserve = self._budget.size(heading) + self._separator_size
        return max(1, self._budget.limit - reserve)

    def _pack_blocks(self, blocks: list[Block], heading: str | None) -> None:
        limit = self._reserved_limit(heading)
        # Small-to-big retrieval: every piece of this oversized section links back
        # to the whole section text, which is what the LLM will actually read.
        parent = self._bound_parent(_SEPARATOR.join(b.text for b in blocks if b.text))
        buffer: list[Block] = []
        buffer_size = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_size
            if not buffer:
                return
            self._emit(
                buffer,
                section=heading,
                sections=[heading] if heading else [],
                prepend_heading=True,
                parent=parent,
            )
            buffer = self._overlap_seed(buffer)
            buffer_size = self._packed_size(buffer)

        for block in blocks:
            text = block.text
            if not text:
                continue

            size = self._budget.size(text)
            if block.block_type == BlockType.TABLE or size > limit:
                flush_buffer()
                buffer, buffer_size = [], 0
                self._emit_oversized(block, text, heading)
                continue

            addition = size + (self._separator_size if buffer else 0)
            if buffer and buffer_size + addition > limit:
                flush_buffer()
                addition = size + (self._separator_size if buffer else 0)
            buffer.append(block)
            buffer_size += addition

        flush_buffer()

    def _packed_size(self, blocks: list[Block]) -> int:
        if not blocks:
            return 0
        sizes = [self._budget.size(b.text) for b in blocks]
        return sum(sizes) + self._separator_size * (len(sizes) - 1)

    def _overlap_seed(self, blocks: list[Block]) -> list[Block]:
        if self._budget.overlap == 0:
            return []
        seed: list[Block] = []
        total = 0
        for block in reversed(blocks):
            size = self._budget.size(block.text) + (self._separator_size if seed else 0)
            if size == 0 or total + size > self._budget.overlap:
                break
            seed.insert(0, block)
            total += size
        if len(seed) == len(blocks):  # never carry the whole chunk forward
            seed = seed[1:]
        return seed

    def _emit_oversized(self, block: Block, text: str, heading: str | None) -> None:
        sections = [heading] if heading else []
        limit = self._reserved_limit(heading)

        if block.block_type == BlockType.TABLE:
            # Keep small tables whole; split large ones by rows so each piece fits
            # the encoder window, repeating the header for standalone meaning.
            pieces = self._split_table(text, limit)
        elif self._budget.size(text) <= limit:
            pieces = [text]
        else:
            pieces = self._split_text(text, limit)

        # Always offer the whole block as parent context; _emit drops it when the
        # piece already is the whole block, and keeps it when the safety net in
        # _enforce_limit splits the block further.
        parent = self._bound_parent(text)
        for piece in pieces:
            self._emit(
                [block],
                section=heading,
                sections=sections,
                text=piece,
                prepend_heading=True,
                parent=parent,
            )

    @staticmethod
    def _bound_parent(text: str) -> str:
        if len(text) <= PARENT_CONTEXT_MAX_CHARS:
            return text
        return text[:PARENT_CONTEXT_MAX_CHARS] + " …"

    def _split_text(self, text: str, limit: int) -> list[str]:
        """Split oversized text on sentence boundaries, with sentence overlap."""
        sentences = _split_sentences(text)
        if not sentences:
            return self._hard_split(text, limit)

        pieces: list[str] = []
        current: list[str] = []
        current_size = 0

        for sentence in sentences:
            size = self._budget.size(sentence)
            if size > limit:
                # A single sentence longer than the budget: flush and hard-split it.
                if current:
                    pieces.append(" ".join(current))
                    current, current_size = [], 0
                pieces.extend(self._hard_split(sentence, limit))
                continue

            if current and current_size + size > limit:
                pieces.append(" ".join(current))
                current, current_size = self._sentence_overlap(current)

            current.append(sentence)
            current_size += size

        if current:
            pieces.append(" ".join(current))
        return pieces

    def _sentence_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        if self._budget.overlap == 0:
            return [], 0
        seed: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            size = self._budget.size(sentence)
            if total + size > self._budget.overlap:
                break
            seed.insert(0, sentence)
            total += size
        if len(seed) == len(sentences):
            seed, total = seed[1:], total - self._budget.size(sentences[0])
        return seed, max(total, 0)

    def _hard_split(self, text: str, limit: int) -> list[str]:
        """Last resort for text with no usable boundary (long words, code, IDs)."""
        words = text.split(" ")
        if len(words) > 1:
            pieces: list[str] = []
            current: list[str] = []
            current_size = 0
            for word in words:
                size = self._budget.size(word)
                if current and current_size + size > limit:
                    pieces.append(" ".join(current))
                    current, current_size = [], 0
                current.append(word)
                current_size += size
            if current:
                pieces.append(" ".join(current))
            if all(self._budget.size(piece) <= limit for piece in pieces):
                return pieces

        # Fall back to fixed windows over characters.
        window = self._char_window(limit)
        step = max(1, window - self._char_overlap())
        return [text[start : start + window] for start in range(0, len(text), step)]

    def _char_window(self, limit: int) -> int:
        # In token mode a token is worth ~3-4 characters; stay conservative.
        return limit if self._budget.unit == "chars" else max(1, limit * 3)

    def _char_overlap(self) -> int:
        return (
            self._budget.overlap
            if self._budget.unit == "chars"
            else max(0, self._budget.overlap * 3)
        )

    def _split_table(self, text: str, limit: int) -> list[str]:
        if self._budget.size(text) <= limit:
            return [text]

        header, rows = self._table_parts(text)
        if not rows:
            # Degenerate export (a header with no data rows, or one merged row):
            # splitting it can only produce header fragments, so keep it whole.
            # The parent context carries the full table to the LLM anyway.
            return [text]

        header_size = (self._budget.size(header) + self._line_size) if header else 0
        # Repeating a header that eats most of the budget leaves no room for data
        # rows — that is exactly how header-only fragments appear in wide tables.
        repeat_header = bool(header) and header_size <= limit * HEADER_REPEAT_MAX_RATIO
        prefix = header if repeat_header else ""
        prefix_size = header_size if repeat_header else 0
        row_limit = max(1, limit - prefix_size)

        pieces: list[str] = []
        current: list[str] = []
        current_size = prefix_size
        for row in rows:
            row_size = self._budget.size(row) + (self._line_size if current else 0)
            # A single very wide row (many columns) does not fit even alone: cut it
            # on cell boundaries so no piece overflows the encoder window.
            if row_size > row_limit:
                if current:
                    pieces.append(self._join_table(prefix, current))
                    current, current_size = [], prefix_size
                for part in self._split_row(row, row_limit):
                    pieces.append(self._join_table(prefix, [part]))
                continue

            if current and current_size + row_size > limit:
                pieces.append(self._join_table(prefix, current))
                current = []
                current_size = prefix_size
            current.append(row)
            current_size += row_size
        if current:
            pieces.append(self._join_table(prefix, current))

        # Name the columns at least once when the header is too big to repeat.
        if pieces and header and not repeat_header:
            first = self._join_table(header, [pieces[0]])
            if self._budget.size(first) <= limit:
                pieces[0] = first
        return pieces or [text]

    def _table_parts(self, text: str) -> tuple[str, list[str]]:
        """Return the Markdown header block and the rows that carry real data."""
        lines = text.split("\n")
        header_lines: list[str] = []
        body = lines
        # A Markdown table header is a row followed by a separator like |---|:--|.
        if len(lines) >= 2 and "|" in lines[0] and set(lines[1].strip()) <= set("|-: "):
            header_lines = lines[:2]
            body = lines[2:]

        rows = [
            row
            for row in body
            # Malformed extractions often carry rows of empty cells; they add
            # tokens but no meaning, so drop them instead of emitting noise.
            if any(cell.strip() for cell in row.split("|")) and not set(row.strip()) <= set("|-: ")
        ]
        return "\n".join(header_lines), rows

    def _split_row(self, row: str, limit: int) -> list[str]:
        cells = [cell for cell in row.split("|") if cell.strip()]
        if len(cells) <= 1:
            return self._hard_split(row, limit)

        parts: list[str] = []
        current: list[str] = []
        current_size = 0
        for cell in cells:
            size = self._budget.size(cell) + 1
            if current and current_size + size > limit:
                parts.append("| " + " | ".join(current) + " |")
                current, current_size = [], 0
            current.append(cell.strip())
            current_size += size
        if current:
            parts.append("| " + " | ".join(current) + " |")
        return parts

    @staticmethod
    def _join_table(header: str, rows: list[str]) -> str:
        body = "\n".join(rows)
        return f"{header}\n{body}" if header else body

    def _emit(
        self,
        blocks: list[Block],
        section: str | None,
        sections: list[str],
        text: str | None = None,
        prepend_heading: bool = False,
        parent: str | None = None,
    ) -> None:
        chunk_text = text if text is not None else _SEPARATOR.join(b.text for b in blocks if b.text)
        # Give continuation chunks of a long section their heading as context, so
        # every chunk is self-describing for retrieval (a "breadcrumb").
        if prepend_heading and section and section not in chunk_text:
            chunk_text = f"{section}\n\n{chunk_text}"

        # Hard guarantee: never emit a chunk the encoder would truncate, whatever
        # the upstream heuristics produced (wide table rows, dense formulas, ...).
        for piece in self._enforce_limit(chunk_text, section if prepend_heading else None):
            # Pieces without a single word or number (table rules, stray glyphs)
            # carry no information and would only dilute the index.
            if not any(char.isalnum() for char in piece):
                continue
            metadata = self._build_metadata(blocks, section, sections)
            # Small-to-big retrieval: the piece is what gets embedded, the parent
            # passage is what the LLM reads (see qa.build_context_from_results).
            if parent and len(parent) > len(piece):
                metadata["context"] = parent
            self._chunks.append(
                Chunk(
                    id=f"{blocks[0].id}-chunk-{self._index}",
                    text=piece,
                    metadata=metadata,
                )
            )
            self._index += 1

    def _enforce_limit(self, text: str, section: str | None = None) -> list[str]:
        if self._budget.size(text) <= self._budget.limit:
            return [text]

        # Keep the breadcrumb on every window, not just the first one.
        prefix = ""
        body = text
        marker = f"{section}{_SEPARATOR}" if section else ""
        if marker and text.startswith(marker):
            prefix, body = marker, text[len(marker) :]

        return self._fit_windows(body, prefix)

    def _fit_windows(self, text: str, prefix: str = "") -> list[str]:
        """Cut text into windows that provably fit the budget, prefix included."""
        pieces: list[str] = []
        start = 0
        while start < len(text):
            window = self._estimate_window(text[start:])
            piece = text[start : start + window]
            # Shrink until the measured size fits: token density varies wildly
            # between prose, references and formulas, so estimates need checking.
            while window > 1 and self._budget.size(prefix + piece) > self._budget.limit:
                window = max(1, int(window * 0.8))
                piece = text[start : start + window]
            pieces.append(prefix + piece)
            start += max(1, window - self._char_overlap())
        return pieces

    def _estimate_window(self, text: str) -> int:
        if self._budget.unit == "chars":
            return self._budget.limit
        sample = text[:2000]
        size = self._budget.size(sample) or 1
        chars_per_unit = max(1.0, len(sample) / size)
        return max(1, int(self._budget.limit * chars_per_unit))

    def _build_metadata(
        self,
        blocks: list[Block],
        section: str | None,
        sections: list[str],
    ) -> dict[str, Any]:
        block_types: list[str] = []
        for block in blocks:
            value = block.block_type.value if block.block_type else block.type
            if value not in block_types:
                block_types.append(value)

        pages = sorted({b.page_number for b in blocks if b.page_number is not None})
        descriptions = [b.vlm_description for b in blocks if b.vlm_description]

        # Preserve useful source coordinates set by parsers (page_number,
        # slide_number, sheet_name, ...); the first block wins on conflicts.
        metadata: dict[str, Any] = {}
        for block in blocks:
            for key, value in block.metadata.items():
                if key not in _SKIP_BLOCK_METADATA:
                    metadata.setdefault(key, value)

        metadata.update(
            {
                "source_file": metadata.get("source_file", self._document.file_name),
                "file_type": self._document.file_type,
                "block_ids": [b.id for b in blocks],
                "block_type": block_types[0] if len(block_types) == 1 else "mixed",
                "block_types": block_types,
            }
        )
        if pages:
            metadata["page_number"] = pages[0]
            metadata["page_numbers"] = pages
        if section:
            metadata["section"] = section
        if sections:
            metadata["sections"] = sections
        if len(blocks) == 1 and blocks[0].bbox is not None:
            metadata["bbox"] = blocks[0].bbox
        if descriptions:
            metadata["vlm_description"] = " ".join(descriptions)

        return metadata
