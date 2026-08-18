"""Formulas and code regions of a PDF, read by the model that already serves us.

A formula has no usable text layer: Docling's layout model finds the region but
leaves it empty, so an 85-page probability lecture reaches the index with
**zero** formulas. Docling can fill those regions with its own vision model
(CodeFormulaV2), and that is correct but slow in a way that hurts: 503 s of the
523 s such a document takes, five regions per forward pass, one document at a
time on the ingestion GPU.

The project already talks to a multimodal model on vLLM, and that server is
built for concurrency. Sending each region as its own small request turns the
same work into wall-clock time that shrinks with the number of requests in
flight — 378 formulas in ~180 s at eight, ~120 s at sixteen — and it reads the
Russian words inside formulas that CodeFormulaV2 turns into control sequences
(``\\i a p { \\i }`` for "Happy"). Measurements and the side-by-side comparison
live in docs/parsing_and_chunking.md.

Being a generative model, it can also fail in ways a specialised model cannot,
so every transcript is validated (:func:`validate_transcript`): refusals,
descriptions of the image instead of its content, decoding loops, and output
cut off mid-formula are rejected, retried once with a larger budget and then
dropped — an empty region is the state we started from, a hallucinated formula
is worse than nothing.

Transcripts are cached by the pixels of the crop, so re-ingesting a corpus (the
normal case while tuning retrieval) pays nothing for formulas at all.

``PDF_ENRICHMENT_ENGINE``: ``auto`` (this path when a VLM endpoint is
configured, Docling's own model otherwise), ``vlm``, ``docling``, ``off``.
"""

import hashlib
import io
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from file_agent.document import Block, BlockType, Document
from file_agent.telemetry import tracer
from file_agent.vlm.base import VLMClient

logger = logging.getLogger(__name__)

ENGINES = ("auto", "vlm", "docling", "off")
DEFAULT_ENGINE = "auto"
# Requests in flight. vLLM batches them server-side; eight keeps the endpoint
# responsive for whatever else is running against it.
DEFAULT_CONCURRENCY = 8
# Crop resolution. CodeFormulaV2 is trained at ~120 dpi; a general model reads
# small sub/superscripts better with more pixels, and a formula crop is tiny.
DEFAULT_DPI = 200
DEFAULT_MAX_TOKENS = 512
# Ingestion budget: a document with more formula regions than this is read up
# to the limit and says so, instead of holding the pipeline for an hour.
DEFAULT_MAX_REGIONS = 1500
# The layout box is tight around the glyphs, and a cut-off subscript is
# unreadable for any model; Docling pads by the same fraction.
EXPANSION = 0.18
# Regions smaller than this many points in either direction are page furniture
# (a stray glyph, a rule), not a formula worth a request.
MIN_REGION_SIDE = 6.0
# Longest transcript we accept for one region; beyond this the model has left
# the formula and started writing the page.
MAX_TRANSCRIPT_CHARS = 2000
# Bump when the prompt or the post-processing changes: cached transcripts of
# the old version are then ignored instead of being served for a different
# question, or in a form the current code would no longer produce.
PROMPT_VERSION = "f2"

FORMULA_PROMPT = (
    "This image is a single formula cut out of a document page.\n"
    "Rules:\n"
    "- Transcribe exactly what is written, as LaTeX. Do not solve, simplify, "
    "rename or complete anything.\n"
    "- Words inside the formula (in any language) stay words: wrap them in "
    "\\text{...} instead of transliterating them.\n"
    "- Several lines of one formula are separated by \\\\.\n"
    "- An equation number in parentheses at the margin becomes \\tag{...}.\n"
    "- If the image holds ordinary text rather than a formula, transcribe the "
    "text as it is.\n"
    "Output only the transcription: no $ delimiters, no code fences, no "
    "explanation."
)

CODE_PROMPT = (
    "This image is a code listing cut out of a document page.\n"
    "Rules:\n"
    "- Transcribe the code exactly, keeping line breaks and indentation.\n"
    "- Do not fix, complete, reformat or comment the code.\n"
    "Output only the code: no code fences and no explanation."
)


def resolve_engine(vlm_available: bool, gpu_available: bool) -> str:
    """Pick the enrichment path for this run.

    ``PDF_ENRICHMENT`` keeps its meaning (``off`` disables enrichment, ``on``
    forces it even where it will be slow); ``PDF_ENRICHMENT_ENGINE`` chooses
    who does the reading. In ``auto`` the remote model wins when it is
    configured — it is faster, and it does not need a GPU on this machine,
    which is what makes formulas available on a laptop at all.
    """
    mode = (os.getenv("PDF_ENRICHMENT") or "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return "off"
    forced = mode in ("1", "true", "yes", "on")

    engine = (os.getenv("PDF_ENRICHMENT_ENGINE") or DEFAULT_ENGINE).strip().lower()
    if engine not in ENGINES:
        raise ValueError(
            f"PDF_ENRICHMENT_ENGINE must be one of {', '.join(ENGINES)}, got {engine!r}"
        )
    if engine == "off":
        return "off"

    if engine in ("auto", "vlm") and vlm_available:
        return "vlm"
    if engine == "vlm" and not vlm_available:
        logger.info("PDF_ENRICHMENT_ENGINE=vlm but no VLM endpoint is configured.")
    if gpu_available or forced:
        return "docling"
    return "off"


@dataclass
class TranscriptCheck:
    ok: bool
    reason: str = ""
    #: True when a larger token budget is likely to fix it.
    retryable: bool = False


@dataclass
class _Region:
    block: Block
    kind: str  # "formula" or "code"
    page: int
    bbox: tuple[float, float, float, float]


class TranscriptCache:
    """Transcripts on disk, keyed by the pixels they were read from."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def get(self, key: str) -> str | None:
        try:
            return self._path(key).read_text(encoding="utf-8")
        except OSError:
            return None

    def put(self, key: str, text: str) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written next to the target and renamed: a crash or a second
            # process must never leave a half-written transcript behind.
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        except OSError:  # a cache that cannot be written must not break parsing
            logger.debug("Could not cache the transcript of %s.", key, exc_info=True)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.txt"


def resolve_cache() -> TranscriptCache | None:
    raw = (os.getenv("PDF_ENRICHMENT_CACHE") or "").strip()
    if raw.lower() in ("off", "0", "false", "no"):
        return None
    if raw:
        return TranscriptCache(Path(raw))
    return TranscriptCache(Path.home() / ".cache" / "file_agent" / "formula_enrichment")


class FormulaEnricher:
    """Fills the formula and code regions Docling left empty."""

    def __init__(
        self,
        vlm_client: VLMClient,
        dpi: int | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.dpi = dpi or int(os.getenv("PDF_ENRICHMENT_DPI", DEFAULT_DPI))
        self.concurrency = max(
            1,
            concurrency or int(os.getenv("PDF_ENRICHMENT_CONCURRENCY", DEFAULT_CONCURRENCY)),
        )
        self.max_tokens = max_tokens or int(
            os.getenv("PDF_ENRICHMENT_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
        self.max_regions = int(os.getenv("PDF_ENRICHMENT_MAX_REGIONS", DEFAULT_MAX_REGIONS))
        self.cache = cache
        self.stats: dict[str, int] = {}
        self._render_lock = threading.Lock()
        self._stats_lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def enrich(self, document: Document, file_path: Path) -> dict[str, int]:
        """Transcribe every pending region of ``document``; drop what stays empty."""
        regions = _pending_regions(document)
        if not regions:
            return {}
        if len(regions) > self.max_regions:
            # A thousand-page book of formulas would otherwise hold ingestion
            # for an hour; say what was skipped instead of quietly truncating.
            logger.warning(
                "%s has %d formula/code regions; reading the first %d "
                "(raise PDF_ENRICHMENT_MAX_REGIONS to read them all).",
                Path(file_path).name,
                len(regions),
                self.max_regions,
            )
            self._count("skipped_over_budget", len(regions) - self.max_regions)
            regions = regions[: self.max_regions]

        path = Path(file_path)
        with tracer.start_as_current_span("file_agent.formula_enrichment") as span:
            span.set_attribute("file_agent.file_name", path.name)
            span.set_attribute("file_agent.region_count", len(regions))

            crops = self._render_regions(path, regions)
            self._count("regions", len(regions))
            self._transcribe_all(crops)
            filled = sum(1 for region in regions if region.block.text.strip())
            span.set_attribute("file_agent.enriched", filled)

        _drop_empty(document)
        stats = dict(sorted(self.stats.items()))
        logger.info(
            "Formula/code enrichment of %s: %d of %d region(s) read %s",
            path.name,
            filled,
            len(regions),
            stats,
        )
        return stats

    # -- rendering ----------------------------------------------------------

    def _render_regions(
        self, path: Path, regions: list[_Region]
    ) -> list[tuple[_Region, Image.Image]]:
        crops: list[tuple[_Region, Image.Image]] = []
        zoom = self.dpi / 72.0
        try:
            pdf = fitz.open(str(path))
        except Exception:
            logger.warning("Could not open %s to crop formulas.", path.name, exc_info=True)
            return crops
        with pdf:
            for region in regions:
                if not (1 <= region.page <= len(pdf)):
                    continue
                image = self._crop(pdf, region, zoom)
                if image is None:
                    self._count("crop_failed")
                    continue
                crops.append((region, image))
        return crops

    def _crop(self, pdf, region: _Region, zoom: float) -> Image.Image | None:
        x0, y0, x1, y1 = region.bbox
        if (x1 - x0) < MIN_REGION_SIDE or (y1 - y0) < MIN_REGION_SIDE:
            self._count("region_too_small")
            return None
        dx, dy = (x1 - x0) * EXPANSION, (y1 - y0) * EXPANSION
        page = pdf[region.page - 1]
        rect = fitz.Rect(x0 - dx, y0 - dy, x1 + dx, y1 + dy) & page.rect
        if rect.is_empty:
            return None
        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
            return Image.open(io.BytesIO(pixmap.tobytes("png")))
        except Exception:
            logger.debug("Could not render region on page %s.", region.page, exc_info=True)
            return None

    # -- transcription ------------------------------------------------------

    def _transcribe_all(self, crops: list[tuple[_Region, Image.Image]]) -> None:
        if not crops:
            return

        # One crop can appear many times in a corpus (the same lemma repeated on
        # every page of a lecture), and the whole document repeats on every
        # re-ingestion, so the work is keyed by the pixels.
        by_key: dict[str, list[_Region]] = {}
        images: dict[str, Image.Image] = {}
        for region, image in crops:
            key = _cache_key(image, region.kind)
            by_key.setdefault(key, []).append(region)
            images[key] = image
        self._count("unique", len(by_key))

        pending = []
        for key, regions in by_key.items():
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                self._count("cached", len(regions))
                _apply(regions, cached, source="cache")
                continue
            pending.append(key)
        if not pending:
            return

        # The first request runs alone: when the endpoint is down or the model
        # is text-only, the document costs one timeout instead of hundreds.
        first, rest = pending[0], pending[1:]
        text = self._safe_transcribe(images[first], by_key[first][0].kind)
        if text is None:
            logger.warning("Formula enrichment: the model did not answer; regions stay empty.")
            self._count("endpoint_unavailable")
            return
        self._store(first, text, by_key, images)

        if not rest:
            return
        workers = min(self.concurrency, len(rest))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(
                lambda key: (key, self._safe_transcribe(images[key], by_key[key][0].kind)), rest
            )
            for key, transcript in results:
                if transcript is None:
                    self._count("request_failed")
                    continue
                self._store(key, transcript, by_key, images)

    def _store(self, key, text, by_key, images) -> None:
        if not text:
            return
        if self.cache:
            self.cache.put(key, text)
        _apply(by_key[key], text, source="vlm")

    def _safe_transcribe(self, image: Image.Image, kind: str) -> str | None:
        """Return a validated transcript, ``""`` when rejected, ``None`` on error."""
        try:
            text, finish = self._ask(image, kind, self.max_tokens)
        except Exception:
            logger.debug("Formula request failed.", exc_info=True)
            return None

        check = validate_transcript(text, kind)
        if check.retryable or finish == "length":
            self._count(f"retry_{check.reason or 'truncated'}")
            try:
                retry_text, _ = self._ask(image, kind, self.max_tokens * 2)
            except Exception:
                logger.debug("Formula retry failed.", exc_info=True)
                retry_text = ""
            retry_check = validate_transcript(retry_text, kind)
            # Keep the retry when it is clean, or when the first answer was not
            # usable at all; a suspicious-but-readable first answer is still
            # better than a second one with the same doubt.
            if retry_check.ok and (not check.ok or not retry_check.retryable):
                text, check = retry_text, retry_check

        if not check.ok:
            self._count(f"rejected_{check.reason}")
            return ""
        self._count("read" if not check.reason else f"read_{check.reason}")
        return clean_transcript(text)

    def _ask(self, image: Image.Image, kind: str, max_tokens: int) -> tuple[str, str | None]:
        prompt = FORMULA_PROMPT if kind == "formula" else CODE_PROMPT
        verbose = getattr(self.vlm_client, "describe_image_verbose", None)
        if callable(verbose):
            return verbose(image, prompt, max_tokens=max_tokens)
        return self.vlm_client.describe_image(image, prompt, max_tokens=max_tokens), None

    def _count(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount


# -- validation ---------------------------------------------------------------

_FENCE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_SPACING_RUN = re.compile(r"(\\(?:qquad|quad|;|:|,|!))(?:\s*\\(?:qquad|quad|;|:|,|!)){2,}")
# Macros that change how a symbol is set, not what it means.
_TYPOGRAPHIC = re.compile(
    r"\\(?:operatorname|mathrm|textrm|textnormal|textsf|textstyle|displaystyle|scriptstyle)"
    r"\s*\{([^{}]*)\}"
)
_REFUSAL = re.compile(
    r"^(i (?:cannot|can't|am unable)|sorry|unable to|as an ai|извин|я не мог|не могу)",
    re.IGNORECASE,
)
# The model answering *about* the picture instead of transcribing it.
_PROSE = re.compile(
    r"^(the (image|formula|picture|equation)\b|this (image|is a|appears)|"
    r"на (изображени|рисунке|картинке)|(это|здесь) (изображ|формула|показан)|"
    r"i see\b|изображена?\b)",
    re.IGNORECASE,
)
_MATH = re.compile(r"[\\^_={}∑∫±≤≥×÷·√∞→αβγδθλμπσφω]|\d\s*[+\-*/=]\s*\d")
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# A line, or a short fragment, repeated this often is a decoding loop.
MAX_REPEATS = 6
MAX_BRACE_IMBALANCE = 6


def clean_transcript(text: str) -> str:
    """Strip the wrappers models add around a transcription."""
    cleaned = _FENCE.sub("", text or "").strip()
    # Models pad a right-aligned equation number with a run of spacing macros
    # ("\qquad \qquad \qquad ..."); they carry nothing and cost tokens in every
    # chunk the formula lands in.
    cleaned = _SPACING_RUN.sub(lambda match: match.group(1), cleaned)
    cleaned = simplify_latex(cleaned)
    for opening, closing in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if (
            cleaned.startswith(opening)
            and cleaned.endswith(closing)
            and len(cleaned) > len(opening) + len(closing)
        ):
            cleaned = cleaned[len(opening) : -len(closing)].strip()
            break
    return cleaned


def simplify_latex(text: str) -> str:
    """Drop the typographic macros a model wraps every symbol in.

    ``\\operatorname { P } ( A \\mid B )`` and ``P ( A \\mid B )`` render the
    same, but the first spends half its tokens on typesetting: the retriever
    embeds those tokens, and BM25 matches them, so the formula competes with
    the prose around it on noise rather than on content.
    """
    simplified = text
    for _ in range(3):  # nested wrappers: \mathrm { \mathrm { P } }
        replaced = _TYPOGRAPHIC.sub(lambda match: match.group(1).strip(), simplified)
        if replaced == simplified:
            break
        simplified = replaced
    return re.sub(r"[ \t]{2,}", " ", simplified).strip()


def validate_transcript(text: str, kind: str = "formula") -> TranscriptCheck:
    """Decide whether a transcription of one region can be indexed."""
    cleaned = clean_transcript(text)
    if not cleaned:
        return TranscriptCheck(False, "empty")
    if _REFUSAL.match(cleaned):
        return TranscriptCheck(False, "refusal")
    if _PROSE.match(cleaned):
        return TranscriptCheck(False, "prose")
    if len(cleaned) > MAX_TRANSCRIPT_CHARS:
        return TranscriptCheck(False, "too_long")
    if _repeats(cleaned) > MAX_REPEATS:
        return TranscriptCheck(False, "loop")
    if kind == "formula":
        imbalance = abs(cleaned.count("{") - cleaned.count("}"))
        if imbalance > MAX_BRACE_IMBALANCE:
            return TranscriptCheck(False, "unbalanced", retryable=True)
        if imbalance:
            # One or two unclosed groups: usually an answer cut off mid-formula,
            # which a larger budget fixes — but a readable formula with a stray
            # brace is still worth indexing if the retry is no better.
            return TranscriptCheck(True, "unbalanced", retryable=True)
        if not _MATH.search(cleaned) and len(_WORD.findall(cleaned)) > 12:
            # No mathematics and a paragraph of words: the model described the
            # page instead of transcribing the region.
            return TranscriptCheck(False, "not_a_formula")
    return TranscriptCheck(True)


def _repeats(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        best, run = 1, 1
        for previous, current in zip(lines, lines[1:], strict=False):
            run = run + 1 if current == previous else 1
            best = max(best, run)
        if best > 1:
            return best
    # Single-line loops ("x + x + x + ..."), measured on a repeated fragment.
    fragment = text[:24].strip()
    if len(fragment) >= 8:
        return text.count(fragment)
    return 1


# -- block plumbing -----------------------------------------------------------

ENRICHMENT_PENDING = "enrichment_pending"


def _pending_regions(document: Document) -> list[_Region]:
    regions: list[_Region] = []
    for block in document.blocks:
        if not block.metadata.get(ENRICHMENT_PENDING):
            continue
        if block.text.strip() or block.bbox is None or block.page_number is None:
            continue
        kind = "code" if block.block_type == BlockType.CODE else "formula"
        regions.append(_Region(block, kind, int(block.page_number), tuple(block.bbox)))
    return regions


def _apply(regions: list[_Region], text: str, source: str) -> None:
    for region in regions:
        region.block.text = text
        region.block.metadata["enrichment"] = source
        region.block.metadata.pop(ENRICHMENT_PENDING, None)


def _drop_empty(document: Document) -> None:
    """Remove the regions nothing could read: an empty block only costs recall."""
    kept: list[Block] = []
    for block in document.blocks:
        if block.metadata.pop(ENRICHMENT_PENDING, None) and not block.text.strip():
            continue
        kept.append(block)
    document.blocks = kept


def _cache_key(image: Image.Image, kind: str) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    digest = hashlib.sha1(buffer.getvalue())  # noqa: S324 - a cache key, not a signature
    digest.update(f"|{kind}|{PROMPT_VERSION}".encode())
    return digest.hexdigest()
