"""Text normalisation for lexical (BM25) matching.

The full-text index tokenises, lowercases and Snowball-stems Russian on its own,
but two things it does not do cost real matches on Russian documents:

* ``ё`` and ``е`` are different characters to it, and both spellings of the same
  word are common (``ещё`` / ``еще``, ``учёт`` / ``учет``): a query written one
  way misses a document written the other way outright.
* stemming is a suffix heuristic, not morphology — ``люди`` and ``человек``,
  ``шёл`` and ``идти``, ``лучше`` and ``хороший`` never meet, and short stems
  collide across unrelated words. Lemmatisation (pymorphy3) maps every form to
  its dictionary head word.

Both sides of the match — the indexed text and the query — must go through the
*same* function, which is why it lives here rather than inside the retriever.
The original text is untouched: normalisation feeds a separate index column.
"""

import logging
import os
import re
import unicodedata
from functools import lru_cache

logger = logging.getLogger(__name__)

# Words (letters/digits/underscore, with inner hyphens or apostrophes kept so
# ``A/B-тест`` and ``don't`` survive as units) and numbers with a decimal part.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)+|\w+(?:[-']\w+)*", re.UNICODE)
_CYRILLIC = re.compile(r"[а-яё]")
_LEMMA_CACHE_SIZE = 262_144
DEFAULT_LEMMATIZE = "auto"


def fold_yo(text: str) -> str:
    """Spell ``ё`` as ``е`` (both cases) so the two spellings match each other."""
    return text.replace("ё", "е").replace("Ё", "Е")


def lemmatization_enabled() -> bool:
    """``BM25_LEMMATIZE``: ``auto`` (default) uses pymorphy3 when it is
    installed, ``on`` requires it, ``off`` keeps ё-folding only."""
    setting = (os.getenv("BM25_LEMMATIZE") or DEFAULT_LEMMATIZE).strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        if _morph() is None:
            raise RuntimeError("BM25_LEMMATIZE=on but pymorphy3 is not installed")
        return True
    return _morph() is not None


def normalize_for_fts(text: str, lemmatize: bool | None = None) -> str:
    """Return the lexical form of ``text`` that the FTS column indexes.

    NFC, lowercase, ``ё`` → ``е``, tokens joined by single spaces, and — when
    lemmatisation is on — every Cyrillic token replaced by its lemma. Latin
    words and numbers pass through lowercased; the index's own English/Russian
    stemmer still runs on top, which is harmless on a lemma.
    """
    if not text:
        return ""
    lemmatize = lemmatization_enabled() if lemmatize is None else lemmatize
    folded = fold_yo(unicodedata.normalize("NFC", text)).lower()
    tokens = _TOKEN.findall(folded)
    if lemmatize:
        tokens = [_lemma(token) if _CYRILLIC.search(token) else token for token in tokens]
    return " ".join(tokens)


@lru_cache(maxsize=_LEMMA_CACHE_SIZE)
def _lemma(token: str) -> str:
    morph = _morph()
    if morph is None:
        return token
    # Hyphenated compounds are lemmatised part by part (``научно-технический``
    # is one dictionary entry, ``онлайн-курсам`` is not).
    if "-" in token:
        return "-".join(_lemma(part) for part in token.split("-") if part)
    try:
        parses = morph.parse(token)
    except Exception:  # pragma: no cover - defensive: never fail indexing on a token
        return token
    if not parses:
        return token
    return fold_yo(parses[0].normal_form)


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3
    except ImportError:
        logger.info("pymorphy3 is not installed; BM25 uses stemming only")
        return None
    try:
        return pymorphy3.MorphAnalyzer()
    except Exception as exc:  # pragma: no cover - dictionaries missing
        logger.warning("pymorphy3 could not load its dictionaries (%s); stemming only", exc)
        return None
