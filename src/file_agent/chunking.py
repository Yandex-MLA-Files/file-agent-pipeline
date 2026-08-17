"""Retrieval chunking.

Two strategies share one public entry point, :func:`chunk_document`:

``structured`` (default)
    Section-coherent packing driven by the typed blocks that the parsers emit,
    with heading-path breadcrumbs, per-block-type splitting (prose by
    sentences, lists by items, tables by rows with the header repeated, code by
    lines) and small-to-big parent context windows. See :class:`_Chunker`.

``legacy``
    The original section-packing chunker, kept verbatim in
    :mod:`file_agent.chunking_legacy` (``CHUNKING_STRATEGY=legacy``).

Both budget chunk size in the retrieval encoder's own tokens when a tokenizer
is available (see :func:`get_embedding_tokenizer`), characters otherwise.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol

from file_agent.document import Block, BlockType, Document, heading_level
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 100
# Upper bound for an auto-detected token budget: most sentence-transformers
# encoders top out at 512 positions, and tokenizers often report a sentinel
# value (e.g. 1e30) as ``model_max_length``.
MAX_AUTO_TOKENS = 512

# Target token budget in token mode when the encoder allows more. Retrieval
# precision is best with focused chunks (~250-400 tokens); the LLM still reads
# the surrounding parent passage, so nothing is lost by keeping pieces small.
DEFAULT_TARGET_TOKENS = 384

_SEPARATOR = "\n\n"

# Per-block parser internals that are meaningless once blocks are packed
# together (and never useful to the LLM or the UI).
_SKIP_BLOCK_METADATA = frozenset(
    {
        "docling_label",
        "docling_parent",
        "docling_parent_label",
        "hierarchy_level",
        "block_type",
        "style",
        "item_count",
        "row_count",
        "figure_index",
        "table_index",
        "synthetic_title",
        "slide_layout",
        "language",
    }
)

# Upper bound for the parent passage stored in chunk metadata (small-to-big
# retrieval): small chunks give precise embeddings, but the LLM answers from the
# surrounding section, so each chunk carries its parent text up to this size.
PARENT_CONTEXT_MAX_CHARS = 4000

# A table header is repeated on every piece only while it stays this small a
# share of the budget; a huge header would crowd out the actual data rows.
HEADER_REPEAT_MAX_RATIO = 0.25

# Row records (multi-representation indexing of tables). A table split into
# row *windows* answers "show me this part of the table", but not "which row
# has FIDE 1260" — the query words and the answer sit in different columns of
# one row, and a window of fifteen rows dilutes them. Every row is therefore
# indexed a second time as a self-describing record ("Column: value; ..."),
# which is what BM25 and the encoder can actually match. The chunk still
# carries the surrounding table as its parent passage, so the LLM reads the
# table, not the record.
DEFAULT_TABLE_ROW_RECORDS = True
# Small tables already fit in one chunk; huge ones would flood the index.
TABLE_ROW_RECORD_MIN_ROWS = 4
TABLE_ROW_RECORD_MAX_ROWS = 600
TABLE_ROW_RECORD_MAX_COLUMNS = 40
# Rows of a table whose cells are prose (a two-column "term/definition" table)
# are already good chunks; records would only duplicate them.
TABLE_ROW_RECORD_MAX_CELL_CHARS = 300

# The breadcrumb ("Doc title > Chapter > Section") prepended to chunk text may
# use at most this share of the budget; deeper crumbs are dropped first.
BREADCRUMB_MAX_RATIO = 0.2
BREADCRUMB_SEPARATOR = " > "
BREADCRUMB_MAX_CRUMB_CHARS = 80

# Sentence boundary for Latin/Cyrillic prose. Abbreviations that end with a
# period but do not close a sentence are protected below.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[^\sa-zа-яё])|\n+")
_ABBREVIATIONS = re.compile(
    r"\b(т|т\.е|т\.к|т\.д|т\.п|др|пр|см|стр|рис|табл|гл|п|пп|ст|г|гг|в|вв|тыс|млн|млрд|руб|коп"
    r"|им|напр|ул|д|корп|e\.g|i\.e|etc|vs|fig|no|approx|dr|mr|mrs|ms|prof|vol|pp)\.\s",
    re.IGNORECASE,
)

ChunkStrategy = Literal["structured", "legacy"]
DEFAULT_CHUNK_STRATEGY: ChunkStrategy = "structured"


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


# -- budget ------------------------------------------------------------------------


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
        self._cache: dict[str, int] = {}

    @property
    def unit(self) -> str:
        return "tokens" if self._tokenizer is not None else "chars"

    def size(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            # verbose=False silences the tokenizer's "sequence longer than the
            # model maximum" notice: measuring long text is exactly the point.
            size = len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))
        except TypeError:  # tokenizers that do not accept those keywords
            try:
                size = len(self._tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                size = len(self._tokenizer.encode(text))
        except (ValueError, RuntimeError, OSError):  # pragma: no cover - tokenizer issues
            size = len(text)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[text] = size
        return size


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

    limit = max_tokens or min(_tokenizer_limit(tokenizer), _target_tokens())
    # Keep the caller's overlap *ratio* when switching units, so the defaults
    # (100 of 1000 chars) stay a sensible 10% in token space too.
    scaled_overlap = round(limit * overlap / max_chars) if max_chars else 0
    return _Budget(
        limit=limit,
        minimum=max(1, limit // 3),
        overlap=max(0, min(scaled_overlap, limit - 1)),
        tokenizer=tokenizer,
    )


def _target_tokens() -> int:
    raw = os.getenv("CHUNK_TARGET_TOKENS")
    if not raw:
        return DEFAULT_TARGET_TOKENS
    try:
        return max(32, int(raw))
    except ValueError:
        return DEFAULT_TARGET_TOKENS


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
    if model_name:
        name = model_name
    else:
        from file_agent.lancedb_retriever import resolve_embedding_model_name

        name = resolve_embedding_model_name()
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


# -- public entry point ---------------------------------------------------------------


def resolve_chunk_strategy(explicit: str | None = None) -> ChunkStrategy:
    value = (explicit or os.getenv("CHUNKING_STRATEGY") or DEFAULT_CHUNK_STRATEGY).strip().lower()
    if value not in ("structured", "legacy"):
        raise ValueError(f"CHUNKING_STRATEGY must be 'structured' or 'legacy', got {value!r}")
    return value  # type: ignore[return-value]


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int | None = None,
    max_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
    strategy: ChunkStrategy | None = None,
) -> list[Chunk]:
    """Split a document into retrieval-sized, section-coherent chunks.

    The structured chunker follows the principles used by production RAG stacks:

    1. **Structure first.** Blocks are grouped into sections (a heading plus its
       body), so a heading always opens a chunk and never dangles at the end of
       an unrelated one. Whole sections are then packed together up to the size
       budget, which keeps small slides/paragraphs from becoming useless
       single-sentence chunks while never mixing a section into a chunk that is
       already large enough to stand on its own.
    2. **Context in every chunk.** Each chunk is prefixed with its heading path
       ("Title > Chapter > Section") so the embedding and BM25 index know where
       the passage sits, and the same path is stored in ``metadata["heading_path"]``.
    3. **Budget in the encoder's unit.** When ``tokenizer`` is provided, chunk
       size is measured in tokens instead of characters, so nothing is silently
       truncated at index time and Russian and English are treated consistently.
    4. **Split by content type.** Prose is cut at sentence boundaries with
       sentence overlap, lists by items, tables by rows repeating the header,
       code by lines; a piece never starts with a dangling heading.
    5. **Small-to-big.** Pieces of a long section carry a parent passage
       (``metadata["context"]``) — a window of the section around the piece —
       which is what the LLM actually reads.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    active = resolve_chunk_strategy(strategy)
    if active == "legacy":
        from file_agent.chunking_legacy import chunk_document_legacy

        return chunk_document_legacy(
            document,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )

    with tracer.start_as_current_span("file_agent.chunk_document") as span:
        span.set_attribute("file_agent.block_count", len(document.blocks))
        span.set_attribute("file_agent.max_chars", max_chars)
        span.set_attribute("file_agent.overlap", overlap)
        span.set_attribute("file_agent.strategy", active)

        budget = _build_budget(max_chars, overlap, min_chars, max_tokens, tokenizer)
        chunker = _Chunker(document, budget)
        sections = _group_sections(document.blocks)
        for section in sections:
            chunker.add_section(section)
        chunks = chunker.finish(sections)

        span.set_attribute("file_agent.chunk_count", len(chunks))
        logger.info("Chunked %d block(s) into %d chunk(s)", len(document.blocks), len(chunks))
        return chunks


# -- sections ------------------------------------------------------------------------


@dataclass
class _Section:
    """A heading and the blocks that belong to it (its reading-order body)."""

    heading: str | None
    blocks: list[Block]
    # Headings above this section, outermost first (excluding its own heading).
    path: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _SEPARATOR.join(b.text for b in self.blocks if b.text)

    @property
    def body_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.block_type != BlockType.HEADING]


def _group_sections(blocks: list[Block]) -> list[_Section]:
    """Split blocks into leaf sections and record the heading path of each."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)
    current = _Section(heading=None, blocks=[], path=[])
    for block in blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            if current.blocks:
                sections.append(current)
            level = heading_level(block)
            while stack and stack[-1][0] >= level:
                stack.pop()
            path = [text for _, text in stack]
            title = " ".join(block.text.split())
            stack.append((level, title))
            current = _Section(heading=title, blocks=[block], path=path)
        else:
            current.blocks.append(block)
    if current.blocks:
        sections.append(current)
    return sections


_NBSP = "\u00a0"


def _split_sentences(text: str) -> list[str]:
    # Glue abbreviations ("т.е. ", "стр. ", "e.g. ") to the next word with a
    # non-breaking space so the boundary regex does not cut a sentence there.
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _NBSP, text)
    parts = [part.replace(_NBSP, " ").strip() for part in _SENTENCE_BOUNDARY.split(protected)]
    return [part for part in parts if part]


# -- chunker -----------------------------------------------------------------------


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
        self._title = _document_title(document)

    def add_section(self, section: _Section) -> None:
        text = section.text
        if not text:
            return

        crumb = self._breadcrumb(section.path)
        size = self._budget.size(text) + self._crumb_size(crumb)
        if size > self._budget.limit:
            self._flush()
            self._pack_blocks(section)
            return

        addition = size + (self._separator_size if self._buffer else 0)
        would_exceed = self._buffer_size + addition > self._budget.limit
        if self._buffer and (self._buffer_size >= self._budget.minimum or would_exceed):
            self._flush()
            addition = size
        # Sections packed together share one breadcrumb (that of the first);
        # a section from a different branch of the outline starts a new chunk.
        if self._buffer and self._buffer[0].path != section.path and self._buffer_size > 0:
            if self._buffer_size >= self._budget.minimum // 2:
                self._flush()
                addition = size

        self._buffer.append(section)
        self._buffer_size += addition

    def finish(self, sections: list[_Section] | None = None) -> list[Chunk]:
        self._flush()
        if sections and _row_records_enabled():
            self._add_table_records(sections)
        return self._chunks

    # -- packing whole sections ------------------------------------------------------

    def _flush(self) -> None:
        if not self._buffer:
            return
        blocks = [block for section in self._buffer for block in section.blocks]
        headings = [section.heading for section in self._buffer if section.heading]
        path = self._buffer[0].path
        self._emit(
            blocks,
            section=headings[0] if headings else (path[-1] if path else None),
            sections=headings,
            path=path,
        )
        self._buffer = []
        self._buffer_size = 0

    # -- splitting oversized sections -------------------------------------------------

    def _pack_blocks(self, section: _Section) -> None:
        heading = section.heading
        # The heading itself travels in the breadcrumb of every piece; the body
        # is what gets packed. A piece never consists of a heading alone.
        blocks = section.body_blocks or section.blocks
        path = section.path + ([heading] if heading else [])
        limit = self._reserved_limit(path)
        buffer: list[Block] = []
        buffer_size = 0
        # Positions of the pieces inside the section, for parent windows.
        cursor = 0

        def flush_buffer(end_index: int) -> None:
            nonlocal buffer, buffer_size
            if not buffer:
                return
            parent = self._parent_window(blocks, end_index - len(buffer), end_index, path)
            self._emit(
                buffer,
                section=heading,
                sections=[heading] if heading else [],
                path=path,
                parent=parent,
            )
            buffer = self._overlap_seed(buffer)
            buffer_size = self._packed_size(buffer)

        for index, block in enumerate(blocks):
            text = block.text
            if not text:
                continue

            size = self._budget.size(text)
            if block.block_type == BlockType.TABLE or size > limit:
                flush_buffer(index)
                buffer, buffer_size = [], 0
                parent = self._parent_window(blocks, index, index + 1, path)
                self._emit_oversized(block, text, heading, path, parent)
                cursor = index + 1
                continue

            addition = size + (self._separator_size if buffer else 0)
            if buffer and buffer_size + addition > limit:
                flush_buffer(index)
                addition = size + (self._separator_size if buffer else 0)
            buffer.append(block)
            buffer_size += addition
            cursor = index + 1

        flush_buffer(cursor)

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

    def _parent_window(self, blocks: list[Block], start: int, end: int, path: list[str]) -> str:
        """Section text around ``blocks[start:end]``, grown both ways up to the cap.

        The heading path opens the passage so the LLM knows which section it is
        reading even when the window starts mid-section.
        """
        start = max(0, start)
        end = max(start, min(end, len(blocks)))
        text = _SEPARATOR.join(b.text for b in blocks[start:end] if b.text)
        heading_line = BREADCRUMB_SEPARATOR.join(path)
        cap = PARENT_CONTEXT_MAX_CHARS - (
            len(heading_line) + len(_SEPARATOR) if heading_line else 0
        )
        if len(text) >= cap:
            text = text[:cap] + " …"
            return f"{heading_line}{_SEPARATOR}{text}" if heading_line else text
        before, after = start - 1, end
        while before >= 0 or after < len(blocks):
            grew = False
            if before >= 0:
                candidate = blocks[before].text
                if not candidate:
                    grew = True
                elif len(text) + len(candidate) + len(_SEPARATOR) <= cap:
                    text = candidate + _SEPARATOR + text
                    grew = True
                before -= 1
            if after < len(blocks):
                candidate = blocks[after].text
                if not candidate:
                    grew = True
                elif len(text) + len(candidate) + len(_SEPARATOR) <= cap:
                    text = text + _SEPARATOR + candidate
                    grew = True
                after += 1
            if not grew:
                break
        return f"{heading_line}{_SEPARATOR}{text}" if heading_line else text

    def _emit_oversized(
        self, block: Block, text: str, heading: str | None, path: list[str], parent: str
    ) -> None:
        sections = [heading] if heading else []
        limit = self._reserved_limit(path)

        if block.block_type == BlockType.TABLE:
            # Keep small tables whole; split large ones by rows so each piece fits
            # the encoder window, repeating the header for standalone meaning.
            pieces = self._split_table(text, limit)
        elif self._budget.size(text) <= limit:
            pieces = [text]
        elif block.block_type == BlockType.LIST:
            pieces = self._split_lines(text, limit, keep_blank_lines=False)
        elif block.block_type == BlockType.CODE:
            pieces = self._split_lines(text, limit, keep_blank_lines=True)
        else:
            pieces = self._split_text(text, limit)

        # Always offer the whole block as parent context; _emit drops it when the
        # piece already is the whole block, and keeps it when the safety net in
        # _enforce_limit splits the block further.
        whole = self._bound_parent(text)
        heading_line = BREADCRUMB_SEPARATOR.join(path)
        if heading_line and len(pieces) > 1:
            whole = f"{heading_line}{_SEPARATOR}{whole}"
        block_parent = whole if len(whole) >= len(parent) or len(pieces) > 1 else parent
        for piece in pieces:
            self._emit(
                [block],
                section=heading,
                sections=sections,
                path=path,
                text=piece,
                parent=block_parent,
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

    def _split_lines(self, text: str, limit: int, keep_blank_lines: bool) -> list[str]:
        """Split lists/code on line boundaries (items are never cut in half)."""
        lines = text.split("\n")
        if not keep_blank_lines:
            lines = [line for line in lines if line.strip()]
        pieces: list[str] = []
        current: list[str] = []
        current_size = 0
        for line in lines:
            size = self._budget.size(line)
            if size > limit:
                if current:
                    pieces.append("\n".join(current))
                    current, current_size = [], 0
                pieces.extend(self._split_text(line, limit))
                continue
            addition = size + (self._line_size if current else 0)
            if current and current_size + addition > limit:
                pieces.append("\n".join(current))
                current, current_size = [], 0
                addition = size
            current.append(line)
            current_size += addition
        if current:
            pieces.append("\n".join(current))
        return pieces or [text]

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

    # -- tables --------------------------------------------------------------------

    def _split_table(self, text: str, limit: int) -> list[str]:
        if self._budget.size(text) <= limit:
            return [text]

        caption, header, rows = self._table_parts(text)
        if not rows:
            # Degenerate export (a header with no data rows, or one merged row):
            # splitting it can only produce header fragments, so keep it whole.
            # The parent context carries the full table to the LLM anyway.
            return [text]

        prefix_lines = [line for line in (caption, header) if line]
        prefix = "\n".join(prefix_lines)
        prefix_size = (self._budget.size(prefix) + self._line_size) if prefix else 0
        # Repeating a header that eats most of the budget leaves no room for data
        # rows — that is exactly how header-only fragments appear in wide tables.
        repeat_prefix = bool(prefix) and prefix_size <= limit * HEADER_REPEAT_MAX_RATIO
        if not repeat_prefix and caption and header:
            # Try the header alone (captions can be long).
            prefix = header
            prefix_size = self._budget.size(prefix) + self._line_size
            repeat_prefix = prefix_size <= limit * HEADER_REPEAT_MAX_RATIO
        if not repeat_prefix:
            prefix, prefix_size = "", 0
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
        if pieces and header and not repeat_prefix:
            first = self._join_table(header, [pieces[0]])
            if self._budget.size(first) <= limit:
                pieces[0] = first
        return pieces or [text]

    def _table_parts(self, text: str) -> tuple[str, str, list[str]]:
        """Return (caption lines, Markdown header block, data rows)."""
        lines = text.split("\n")
        # Non-table lines before the first pipe row are the caption/title.
        first_row = 0
        while first_row < len(lines) and "|" not in lines[first_row]:
            first_row += 1
        caption = "\n".join(line for line in lines[:first_row] if line.strip())
        lines = lines[first_row:]

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
        return caption, "\n".join(header_lines), rows

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

    # -- table row records ----------------------------------------------------------

    def _add_table_records(self, sections: list[_Section]) -> None:
        for section in sections:
            heading = section.heading
            path = section.path + ([heading] if heading else [])
            for block in section.blocks:
                if block.block_type == BlockType.TABLE and block.text:
                    self._emit_table_records(block, heading, path)

    def _emit_table_records(self, block: Block, heading: str | None, path: list[str]) -> None:
        caption, header, rows = self._table_parts(block.text)
        if not header or not (TABLE_ROW_RECORD_MIN_ROWS <= len(rows) <= TABLE_ROW_RECORD_MAX_ROWS):
            return
        columns = _table_cells(header.split("\n")[0])
        if not columns or len(columns) > TABLE_ROW_RECORD_MAX_COLUMNS:
            return
        title = " ".join(caption.split()) if caption else str(block.metadata.get("caption") or "")
        title = _shorten(title, BREADCRUMB_MAX_CRUMB_CHARS * 2) if title else ""

        for index, row in enumerate(rows):
            cells = _table_cells(row)
            if not cells or max((len(cell) for cell in cells), default=0) > (
                TABLE_ROW_RECORD_MAX_CELL_CHARS
            ):
                continue
            pairs = [
                f"{column}: {value}"
                for column, value in zip(columns, cells, strict=False)
                if value.strip() and column.strip()
            ]
            # A record needs at least a key and a value to be worth indexing;
            # a single-cell row carries no relation to retrieve.
            if len(pairs) < 2:
                continue
            record = "; ".join(pairs)
            text = f"{title}\n{record}" if title else record
            self._emit(
                [block],
                section=heading,
                sections=[heading] if heading else [],
                path=path,
                text=text,
                parent=self._row_context(caption, header, rows, index),
                extra={"representation": "row", "row_index": index + 1},
            )

    def _row_context(self, caption: str, header: str, rows: list[str], index: int) -> str:
        """The table around one row: caption, header and as many neighbours as fit."""
        head = "\n".join(line for line in (caption, header) if line)
        budget = PARENT_CONTEXT_MAX_CHARS - len(head)
        window = [rows[index]]
        size = len(rows[index])
        before, after = index - 1, index + 1
        while before >= 0 or after < len(rows):
            grew = False
            if before >= 0 and size + len(rows[before]) + 1 <= budget:
                window.insert(0, rows[before])
                size += len(rows[before]) + 1
                before -= 1
                grew = True
            if after < len(rows) and size + len(rows[after]) + 1 <= budget:
                window.append(rows[after])
                size += len(rows[after]) + 1
                after += 1
                grew = True
            if not grew:
                break
        return self._join_table(head, window)

    # -- breadcrumbs ----------------------------------------------------------------

    def _breadcrumb(self, path: list[str]) -> str:
        """Heading path (with the document title) that prefixes a chunk."""
        crumbs: list[str] = []
        if self._title:
            crumbs.append(self._title)
        for crumb in path:
            if crumb and (not crumbs or crumbs[-1] != crumb):
                crumbs.append(crumb)
        crumbs = [_shorten(c, BREADCRUMB_MAX_CRUMB_CHARS) for c in crumbs]
        max_size = max(1, int(self._budget.limit * BREADCRUMB_MAX_RATIO))
        # Drop the outermost crumbs first: the nearest headings matter most.
        while crumbs and self._budget.size(BREADCRUMB_SEPARATOR.join(crumbs)) > max_size:
            crumbs.pop(0)
        return BREADCRUMB_SEPARATOR.join(crumbs)

    def _crumb_size(self, crumb: str) -> int:
        return (self._budget.size(crumb) + self._separator_size) if crumb else 0

    def _reserved_limit(self, path: list[str]) -> int:
        """Budget available for content once the breadcrumb is added."""
        crumb = self._breadcrumb(path)
        return max(1, self._budget.limit - self._crumb_size(crumb))

    # -- emission -------------------------------------------------------------------

    def _emit(
        self,
        blocks: list[Block],
        section: str | None,
        sections: list[str],
        path: list[str],
        text: str | None = None,
        parent: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        body = text if text is not None else _SEPARATOR.join(b.text for b in blocks if b.text)
        if not any(char.isalnum() for char in body):
            return
        crumb = self._breadcrumb(path)
        # The breadcrumb makes every chunk self-describing for retrieval; when the
        # chunk already opens with that heading, do not repeat it.
        prefix = ""
        if crumb:
            crumbs = crumb.split(BREADCRUMB_SEPARATOR)
            first_line = body.split("\n", 1)[0].strip()
            if crumbs and first_line == crumbs[-1]:
                crumbs = crumbs[:-1]
            if crumbs:
                prefix = f"{BREADCRUMB_SEPARATOR.join(crumbs)}{_SEPARATOR}"
        chunk_text = f"{prefix}{body}"

        # Hard guarantee: never emit a chunk the encoder would truncate, whatever
        # the upstream heuristics produced (wide table rows, dense formulas, ...).
        for piece in self._enforce_limit(chunk_text, prefix):
            # Pieces without a single word or number (table rules, stray glyphs)
            # carry no information and would only dilute the index.
            if not any(char.isalnum() for char in piece):
                continue
            metadata = self._build_metadata(blocks, section, sections, path)
            if extra:
                metadata.update(extra)
            # Small-to-big retrieval: the piece is what gets embedded, the parent
            # passage is what the LLM reads (see qa.build_context_from_results).
            body_only = piece[len(prefix) :] if prefix and piece.startswith(prefix) else piece
            if parent and len(parent) > len(body_only) and parent != body_only:
                metadata["context"] = parent
            self._chunks.append(
                Chunk(
                    id=f"{blocks[0].id}-chunk-{self._index}",
                    text=piece,
                    metadata=metadata,
                )
            )
            self._index += 1

    def _enforce_limit(self, text: str, prefix: str = "") -> list[str]:
        if self._budget.size(text) <= self._budget.limit:
            return [text]

        # Keep the breadcrumb on every window, not just the first one.
        body = text[len(prefix) :] if prefix and text.startswith(prefix) else text
        return self._fit_windows(body, prefix if text.startswith(prefix) else "")

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
            # Always advance by at least half a window: with a large overlap and
            # a token-dense window (dot leaders, hashes) the naive step used to
            # collapse to a single character and produce thousands of chunks.
            step = max(window // 2, window - self._char_overlap(), 1)
            start += step
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
        path: list[str],
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
                if key in _SKIP_BLOCK_METADATA or key.startswith("_"):
                    continue
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
        if self._title:
            metadata["doc_title"] = self._title
        if pages:
            metadata["page_number"] = pages[0]
            metadata["page_numbers"] = pages
        if section:
            metadata["section"] = section
        if sections:
            metadata["sections"] = sections
        full_path = list(path)
        if section and (not full_path or full_path[-1] != section):
            full_path.append(section)
        if full_path:
            metadata["heading_path"] = full_path
        if len(blocks) == 1 and blocks[0].bbox is not None:
            metadata["bbox"] = blocks[0].bbox
        if descriptions:
            metadata["vlm_description"] = " ".join(descriptions)

        return metadata


def _row_records_enabled() -> bool:
    raw = os.getenv("TABLE_ROW_RECORDS")
    if raw is None:
        return DEFAULT_TABLE_ROW_RECORDS
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _table_cells(row: str) -> list[str]:
    """Cells of one Markdown table row, without the outer pipes."""
    stripped = row.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _document_title(document: Document) -> str | None:
    title = document.title
    if not title:
        return None
    return " ".join(title.split())


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
