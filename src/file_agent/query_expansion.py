"""Multi-query expansion: several formulations of one question for retrieval.

A question and the passage that answers it rarely share wording — the user asks
"сколько заработала компания", the report says "выручка составила". Dense
retrieval bridges some of that, BM25 none of it. Asking the answering model for
a few alternative formulations and searching with all of them, then fusing the
result lists, is the cheapest known way to widen recall without touching the
index.

Three ways to expand, chosen with ``MULTI_QUERY_MODE``:

* ``paraphrase`` — N rewordings of the question, same intent, different words;
* ``hyde`` — one hypothetical passage written as if it answered the question
  (HyDE): its embedding sits where the real answer sits, not where the question
  sits, which is what a short factual question needs;
* ``mixed`` — N−1 rewordings plus one hypothetical passage.

Expansions are cached on disk by question, mode, count, model and prompt
version, so a re-run of a benchmark or a dataset produces the same queries and
costs nothing; the model is asked once per distinct question.
"""

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from file_agent.llm.base import LLMClient
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

PROMPT_VERSION = "q1"
DEFAULT_ENABLED = False
DEFAULT_MODE = "paraphrase"
DEFAULT_COUNT = 3
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "file_agent" / "multi_query"
MODES = ("paraphrase", "hyde", "mixed")
MAX_COUNT = 8
MAX_VARIANT_CHARS = 600

QueryExpander = Callable[[str], list[str]]

_PARAPHRASE_PROMPT = """You rewrite search queries for a document retrieval system.

Write {count} alternative formulations of the question below. Each must keep the
same meaning and intent but use different wording: synonyms, the terms a
document would use instead of the terms a person asking would use, a more
specific or a more general phrasing, an expanded abbreviation. Stay in the
language of the question. Do not answer the question. Do not add information
that is not in it.

Output exactly {count} lines, one formulation per line, no numbering, no quotes,
no explanations.

Question: {question}"""

_HYDE_PROMPT = """You help a document retrieval system find the passage that answers a question.

Write one short passage (2-4 sentences) in the language of the question, in the
style of the document that would contain the answer — a report, a lecture, a
manual, a paper — as if it were quoting that document. Use the vocabulary such a
document would use. If you do not know the facts, write plausible placeholder
content in the right form; the exact facts do not matter, the wording does.

Output only the passage, no preamble, no quotes.

Question: {question}"""


@dataclass(frozen=True)
class MultiQuerySettings:
    enabled: bool
    mode: str
    count: int
    cache_dir: Path | None

    @classmethod
    def from_env(cls) -> "MultiQuerySettings":
        enabled = _bool_env("MULTI_QUERY", DEFAULT_ENABLED)
        mode = (os.getenv("MULTI_QUERY_MODE") or DEFAULT_MODE).strip().lower()
        if mode not in MODES:
            raise ValueError(f"MULTI_QUERY_MODE must be one of {', '.join(MODES)}, got {mode!r}")
        count = _int_env("MULTI_QUERY_COUNT", DEFAULT_COUNT)
        if not 1 <= count <= MAX_COUNT:
            raise ValueError(f"MULTI_QUERY_COUNT must be between 1 and {MAX_COUNT}, got {count}")
        raw_cache = (os.getenv("MULTI_QUERY_CACHE") or "").strip()
        if raw_cache.lower() in {"off", "0", "false", "no", "none"}:
            cache_dir: Path | None = None
        else:
            cache_dir = Path(raw_cache).expanduser() if raw_cache else DEFAULT_CACHE_DIR
        return cls(
            enabled=enabled,
            mode=mode,
            count=count,
            cache_dir=cache_dir,
        )


class MultiQueryExpander:
    """Turn one question into a few alternative retrieval queries.

    ``__call__`` never raises: a model failure logs a warning and returns no
    variants, so retrieval degrades to the single original query.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        mode: str = DEFAULT_MODE,
        count: int = DEFAULT_COUNT,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        if not 1 <= count <= MAX_COUNT:
            raise ValueError(f"count must be between 1 and {MAX_COUNT}, got {count}")
        self.llm_client = llm_client
        self.mode = mode
        self.count = count
        self.cache_dir = cache_dir

    def __call__(self, question: str) -> list[str]:
        question = " ".join(question.split())
        if not question:
            return []
        with tracer.start_as_current_span("file_agent.query_expansion") as span:
            span.set_attribute("file_agent.mode", self.mode)
            span.set_attribute("file_agent.count", self.count)
            cached = self._read_cache(question)
            if cached is not None:
                span.set_attribute("file_agent.cache_hit", True)
                return cached
            try:
                variants = self._expand(question)
            except Exception as exc:  # the retriever must keep working without us
                logger.warning("Query expansion failed (%s); searching with the question only", exc)
                span.set_attribute("file_agent.error", str(exc))
                return []
            span.set_attribute("file_agent.variant_count", len(variants))
            self._write_cache(question, variants)
            return variants

    def _expand(self, question: str) -> list[str]:
        variants: list[str] = []
        paraphrases = self.count if self.mode == "paraphrase" else self.count - 1
        if paraphrases > 0:
            raw = self._generate(_PARAPHRASE_PROMPT.format(count=paraphrases, question=question))
            variants.extend(parse_variants(raw, question, limit=paraphrases))
        if self.mode in {"hyde", "mixed"}:
            raw = self._generate(_HYDE_PROMPT.format(question=question))
            passage = " ".join(raw.split())
            if passage and _distinct(passage, question, variants):
                variants.append(passage[:MAX_VARIANT_CHARS])
        return variants[: self.count]

    def _generate(self, prompt: str) -> str:
        return self.llm_client.generate(prompt)

    def _cache_path(self, question: str) -> Path | None:
        if self.cache_dir is None:
            return None
        model = str(getattr(self.llm_client, "model", "") or "")
        key = "\n".join((PROMPT_VERSION, self.mode, str(self.count), model, question))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _read_cache(self, question: str) -> list[str] | None:
        path = self._cache_path(question)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        variants = payload.get("variants")
        if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
            return None
        return variants

    def _write_cache(self, question: str, variants: list[str]) -> None:
        path = self._cache_path(question)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"question": question, "mode": self.mode, "variants": variants}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - read-only cache dir
            logger.debug("Could not write query expansion cache %s: %s", path, exc)


_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-zA-Zа-яА-Я][.)])\s*")
_QUOTES = "\"'«»“”„‘’`"


def parse_variants(raw: str, question: str, limit: int) -> list[str]:
    """Extract clean, distinct formulations from the model's line-per-variant answer."""
    variants: list[str] = []
    for line in raw.splitlines():
        candidate = _LINE_PREFIX.sub("", line).strip().strip(_QUOTES).strip()
        candidate = " ".join(candidate.split())
        if not candidate or len(candidate) > MAX_VARIANT_CHARS:
            continue
        lowered = candidate.lower()
        # Chatter rather than a query: "Here are three formulations:".
        if lowered.endswith(":") or lowered.startswith(("here are", "вот ")):
            continue
        if _distinct(candidate, question, variants):
            variants.append(candidate)
        if len(variants) >= limit:
            break
    return variants


def _distinct(candidate: str, question: str, existing: list[str]) -> bool:
    key = _comparison_key(candidate)
    if not key or key == _comparison_key(question):
        return False
    return all(key != _comparison_key(other) for other in existing)


def _comparison_key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def resolve_query_expander(settings: MultiQuerySettings | None = None) -> QueryExpander | None:
    """The expander the retriever uses when none is injected: from the environment.

    Returns ``None`` when ``MULTI_QUERY`` is off or no LLM endpoint is
    configured, so retrieval silently stays single-query.
    """
    settings = settings or MultiQuerySettings.from_env()
    if not settings.enabled:
        return None
    return _build_expander(settings.mode, settings.count, settings.cache_dir)


@lru_cache(maxsize=4)
def _build_expander(mode: str, count: int, cache_dir: Path | None) -> QueryExpander | None:
    from file_agent.llm.factory import create_llm_client

    try:
        client = create_llm_client()
    except Exception as exc:
        logger.warning("MULTI_QUERY is on but no LLM client could be created (%s)", exc)
        return None
    return MultiQueryExpander(client, mode=mode, count=count, cache_dir=cache_dir)


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
