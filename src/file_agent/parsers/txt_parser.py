"""Structure-aware plain-text parser.

Plain text has no markup, but real files still have structure that matters
for retrieval:

- **Prose / books** — hard-wrapped lines with indented or blank-line separated
  paragraphs, chapter titles as ``* CAPS *`` / ALL-CAPS / short standalone
  lines. Paragraphs are re-flowed into single blocks and titles become
  ``HEADING`` blocks so chunks stay section-coherent.
- **Transcripts** — lines prefixed with ``[mm:ss]`` / ``hh:mm:ss`` timestamps
  (YouTube/meeting exports). Timestamps are stripped from the text (they eat
  most of a small embedding window) and captions are re-flowed into
  paragraphs of ~``TRANSCRIPT_WORDS_PER_BLOCK`` words; each block keeps the
  timestamp of its first caption in metadata so answers can still cite it. A
  leading ``Key: value`` metadata header (title, channel, ...) is kept as text.
- Everything else falls back to blank-line paragraphs.

Encoding is detected (UTF-8/UTF-16, then cp1251/koi8-r/cp866 for legacy
Russian files) so mojibake never reaches the index.
"""

import logging
import re
from pathlib import Path

from file_agent.document import BlockType, Document
from file_agent.parsers.base import BaseParser
from file_agent.parsers.common import (
    BlockFactory,
    infer_heading_level,
    read_text_file,
)
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

TRANSCRIPT_WORDS_PER_BLOCK = 140

_TIMESTAMP = re.compile(r"^\s*\[?(\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?\]?\s*(-->.*)?\s*")
_RULE = re.compile(r"^\s*[-=_*~]{4,}\s*$")
_STARRED_TITLE = re.compile(r"^\s*\*+\s*(.+?)\s*\*+\s*$")
_KEY_VALUE = re.compile(r"^\s*[^\w\s]{0,3}\s*[\w .()/-]{2,40}:\s+\S")
_LETTERS = re.compile(r"[^\W\d_]")
_SENTENCE_END = re.compile(r"[.!?…]\s*$")


class TXTParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        with tracer.start_as_current_span("file_agent.txt_parse") as span:
            span.set_attribute("file_agent.file_name", path.name)

            text = read_text_file(path)
            lines = clean_text_lines(text)
            factory = BlockFactory(source_file=path.name)

            if _is_transcript(lines):
                title = _parse_transcript(lines, factory)
                method = "text-transcript"
            else:
                title = _parse_prose(lines, factory)
                method = "text-prose"

            if not factory.blocks:
                factory.add(text.strip(), BlockType.TEXT, skip_empty=False)

            document = Document(
                file_name=path.name,
                file_type="txt",
                blocks=factory.blocks,
                metadata={"parsing_method": method, "title": title},
            )
            document.build_table_of_contents()
            span.set_attribute("file_agent.block_count", len(document.blocks))
            logger.info("Parsed %s into %d block(s)", path.name, len(document.blocks))
            return document


def clean_text_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return [line.rstrip() for line in text.split("\n")]


# -- transcripts ------------------------------------------------------------------


def _is_transcript(lines: list[str]) -> bool:
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 10:
        return False
    stamped = sum(1 for line in non_empty if _TIMESTAMP.match(line))
    return stamped / len(non_empty) >= 0.6


def _parse_transcript(lines: list[str], factory: BlockFactory) -> str | None:
    title = None
    header: list[str] = []
    captions: list[tuple[str, str]] = []  # (timestamp, text)

    for line in lines:
        if not line.strip():
            continue
        match = _TIMESTAMP.match(line)
        if match:
            stamp = match.group(0).strip().strip("[]").split("-->")[0].strip()
            body = line[match.end() :].strip()
            if body:
                captions.append((stamp, body))
            continue
        if not captions:
            header.append(line.strip())

    if header:
        cleaned = [_strip_emoji_prefix(line) for line in header]
        for line in cleaned:
            lowered = line.lower()
            if lowered.startswith(("title:", "название:", "тема:")):
                title = line.split(":", 1)[1].strip()
                break
        if title:
            factory.heading(title, 1)
        factory.add("\n".join(cleaned), BlockType.TEXT, {"transcript_header": True})

    block_words: list[str] = []
    block_start: str | None = None
    for stamp, body in captions:
        if block_start is None:
            block_start = stamp
        block_words.extend(body.split())
        text = " ".join(block_words)
        # Cut at a sentence end once the target size is reached, or hard-cut
        # when captions carry no punctuation at all (auto-generated subtitles).
        if len(block_words) >= TRANSCRIPT_WORDS_PER_BLOCK and (
            _SENTENCE_END.search(text) or len(block_words) >= TRANSCRIPT_WORDS_PER_BLOCK * 1.5
        ):
            factory.add(text, BlockType.TEXT, {"time_start": block_start})
            block_words, block_start = [], None
    if block_words:
        factory.add(" ".join(block_words), BlockType.TEXT, {"time_start": block_start})
    return title


def _strip_emoji_prefix(line: str) -> str:
    # "📺 Title: ..." -> "Title: ..."
    stripped = line.lstrip()
    while stripped and not (stripped[0].isalnum() or stripped[0] in "[(\"'«"):
        stripped = stripped[1:].lstrip()
    return stripped or line


# -- prose ------------------------------------------------------------------------


def _parse_prose(lines: list[str], factory: BlockFactory) -> str | None:
    title: str | None = None
    paragraph: list[str] = []
    seen_body = False

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(" ".join(paragraph).split())
            if text:
                factory.add(text, BlockType.TEXT)
        paragraph = []

    total = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _RULE.match(stripped):
            flush()
            continue

        heading = _heading_candidate(lines, index, stripped)
        if heading is not None:
            flush()
            text, level = heading
            if title is None and (level == 1 or not seen_body):
                title = text
            factory.heading(text, level)
            continue

        seen_body = True
        indented = len(line) - len(line.lstrip()) >= 2
        # Hard-wrapped prose: an indented line opens a new paragraph when the
        # previous line was flush-left body text.
        if indented and paragraph and not _is_indented(lines[index - 1]) and index > 0:
            flush()
        paragraph.append(stripped)
        if index + 1 >= total:
            flush()
    flush()
    return title


def _is_indented(line: str) -> bool:
    return bool(line.strip()) and len(line) - len(line.lstrip()) >= 2


def _heading_candidate(lines: list[str], index: int, stripped: str) -> tuple[str, int] | None:
    starred = _STARRED_TITLE.match(stripped)
    if starred and _LETTERS.search(starred.group(1)):
        return " ".join(starred.group(1).split()), 1

    if len(stripped) > 90 or _SENTENCE_END.search(stripped) and not stripped.endswith(":"):
        numbered = None
    else:
        numbered = infer_heading_level(stripped)

    letters = _LETTERS.findall(stripped)
    if letters and len(letters) >= 3 and stripped.upper() == stripped and len(stripped) <= 90:
        if not stripped.endswith((".", ",", ";")) and _standalone(lines, index):
            return stripped, 1 if numbered is None else numbered

    if numbered is not None and _standalone(lines, index) and len(stripped.split()) <= 12:
        return stripped, numbered

    # Short standalone line followed by paragraph text (e.g. a myth's name).
    if (
        len(stripped) <= 60
        and len(stripped.split()) <= 6
        and not stripped[0].islower()
        and not stripped.endswith((".", ",", ";", ":", "!", "?"))
        and _LETTERS.search(stripped)
        and not _KEY_VALUE.match(stripped)
        and _standalone(lines, index, require_following_body=True)
    ):
        return stripped, 2
    return None


def _standalone(lines: list[str], index: int, require_following_body: bool = False) -> bool:
    """A heading line is preceded by a break and followed by body text."""
    previous = lines[index - 1] if index > 0 else ""
    following = lines[index + 1] if index + 1 < len(lines) else ""
    preceded_by_break = (
        not previous.strip()
        or _RULE.match(previous)
        or _STARRED_TITLE.match(previous)
        or (previous.strip() and _heading_like(previous))
        # Hard-wrapped books put the next title right after the last sentence
        # of a paragraph; the paragraph that follows starts indented.
        or (bool(_SENTENCE_END.search(previous)) and _is_indented(following))
    )
    if not preceded_by_break:
        return False
    if not following.strip():
        return not require_following_body or bool(_next_non_empty(lines, index + 1))
    return len(following.strip()) > 40 or _is_indented(following)


def _heading_like(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) <= 60 and not _SENTENCE_END.search(stripped)


def _next_non_empty(lines: list[str], start: int) -> str:
    for line in lines[start : start + 3]:
        if line.strip():
            return line
    return ""
