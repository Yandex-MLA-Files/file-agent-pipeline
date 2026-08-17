"""Resolve Word's automatic list numbering (``numbering.xml``).

A numbered list in Word stores no numbers: the paragraph carries a ``numId``
and an indent level, and the visible "1.", "1.2.", "a)" is computed by the
renderer from the numbering definitions. A parser that ignores this either
drops the numbering entirely — turning "8. Унифицированная система …" into an
unnumbered item, so a question about "пункт 8" cannot be answered — or guesses
it from stray digits in the text.

This module reads the definitions and replays the same counting Word does:

* ``numId`` → ``abstractNumId`` (with per-instance ``startOverride``);
* each level has a format (``decimal``, ``lowerLetter``, ``upperRoman``,
  ``bullet``, …) and a template such as ``%1.%2)`` that composes the numbers
  of the enclosing levels;
* counters restart at deeper levels whenever a shallower level advances, and
  a level restarts when its list is entered again.

Everything degrades quietly: a document without ``numbering.xml``, or with
definitions this module cannot read, simply yields no number and the parser
falls back to the previous behaviour.
"""

import logging
import re
from typing import Any

from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

# Word's bullet glyphs live in symbol fonts; they carry no meaning in text.
_BULLET_FORMATS = frozenset({"bullet", "none"})
_LEVEL_PLACEHOLDER = re.compile(r"%(\d)")
_ROMAN = (
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
)


class DocxNumbering:
    """Numbering definitions of one document, plus the running counters."""

    def __init__(self, numbering_element: Any | None) -> None:
        # (abstract id, level) -> {"format": str, "text": str, "start": int}
        self._levels: dict[tuple[str, int], dict[str, Any]] = {}
        # num id -> abstract id, and per-instance start overrides
        self._abstract_of: dict[str, str] = {}
        self._overrides: dict[tuple[str, int], int] = {}
        self._counters: dict[tuple[str, int], int] = {}
        self._seen: set[str] = set()
        if numbering_element is not None:
            self._load(numbering_element)

    @classmethod
    def from_document(cls, docx: Any) -> "DocxNumbering":
        try:
            part = docx.part.numbering_part
        except Exception:
            # A document with no numbered list has no numbering part at all,
            # and python-docx answers that with NotImplementedError rather than
            # a lookup error. Nothing here may ever break parsing.
            return cls(None)
        return cls(getattr(part, "element", None))

    @property
    def available(self) -> bool:
        return bool(self._levels)

    # -- loading ------------------------------------------------------------

    def _load(self, root: Any) -> None:
        try:
            for abstract in root.iter(qn("w:abstractNum")):
                abstract_id = abstract.get(qn("w:abstractNumId"))
                if abstract_id is None:
                    continue
                for level in abstract.iter(qn("w:lvl")):
                    try:
                        ilvl = int(level.get(qn("w:ilvl")))
                    except (TypeError, ValueError):
                        continue
                    self._levels[(abstract_id, ilvl)] = {
                        "format": _child_val(level, "w:numFmt") or "decimal",
                        "text": _child_val(level, "w:lvlText") or "",
                        "start": _child_int(level, "w:start", default=1),
                    }
            for num in root.iter(qn("w:num")):
                num_id = num.get(qn("w:numId"))
                abstract_id = _child_val(num, "w:abstractNumId")
                if num_id is None or abstract_id is None:
                    continue
                self._abstract_of[num_id] = abstract_id
                for override in num.iter(qn("w:lvlOverride")):
                    try:
                        ilvl = int(override.get(qn("w:ilvl")))
                    except (TypeError, ValueError):
                        continue
                    start = _child_int(override, "w:startOverride", default=None)
                    if start is not None:
                        self._overrides[(num_id, ilvl)] = start
        except Exception:  # pragma: no cover - malformed numbering.xml
            logger.debug("Could not read numbering.xml; list numbers will be omitted.")
            self._levels.clear()

    # -- counting -----------------------------------------------------------

    def marker(self, num_id: str | None, ilvl: int) -> str:
        """Advance the counters for one list paragraph and return its marker.

        Returns an empty string for bullets and for anything this module cannot
        resolve — the caller then renders a plain bullet item.
        """
        if not num_id or not self._levels:
            return ""
        abstract_id = self._abstract_of.get(num_id)
        if abstract_id is None:
            return ""
        definition = self._levels.get((abstract_id, ilvl))
        if definition is None:
            return ""

        self._advance(num_id, abstract_id, ilvl)
        if definition["format"] in _BULLET_FORMATS:
            return ""
        return self._render(num_id, abstract_id, ilvl, definition["text"])

    def _advance(self, num_id: str, abstract_id: str, ilvl: int) -> None:
        key = (num_id, ilvl)
        if key in self._counters:
            self._counters[key] += 1
        else:
            self._counters[key] = self._start(num_id, abstract_id, ilvl)
        # A new item at this level restarts everything nested under it, which is
        # what makes "1.1, 1.2, 2.1" come out right instead of "1.1, 1.2, 2.3".
        for other_num, other_level in list(self._counters):
            if other_num == num_id and other_level > ilvl:
                del self._counters[other_num, other_level]
        self._seen.add(num_id)

    def _start(self, num_id: str, abstract_id: str, ilvl: int) -> int:
        override = self._overrides.get((num_id, ilvl))
        if override is not None:
            return override
        return int(self._levels[(abstract_id, ilvl)]["start"])

    def _render(self, num_id: str, abstract_id: str, ilvl: int, template: str) -> str:
        if not template:
            return ""

        def substitute(match: re.Match[str]) -> str:
            level = int(match.group(1)) - 1
            if level > ilvl:
                return ""
            definition = self._levels.get((abstract_id, level))
            if definition is None:
                return ""
            value = self._counters.get((num_id, level))
            if value is None:
                value = self._start(num_id, abstract_id, level)
            return _format_number(value, definition["format"])

        return _LEVEL_PLACEHOLDER.sub(substitute, template).strip()


def _child_val(element: Any, tag: str) -> str | None:
    child = element.find(qn(tag))
    if child is None:
        return None
    return child.get(qn("w:val"))


def _child_int(element: Any, tag: str, default: int | None = 1) -> int | None:
    value = _child_val(element, tag)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: int, number_format: str) -> str:
    fmt = (number_format or "decimal").lower()
    if fmt in ("lowerletter", "upperletter"):
        letters = _to_letters(value)
        return letters.upper() if fmt == "upperletter" else letters
    if fmt in ("lowerroman", "upperroman"):
        roman = _to_roman(value)
        return roman.upper() if fmt == "upperroman" else roman
    if fmt == "decimalzero":
        return f"{value:02d}"
    return str(value)


def _to_letters(value: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa (Word's alphabetic numbering)."""
    if value < 1:
        return ""
    letters = ""
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return letters


def _to_roman(value: int) -> str:
    if value < 1 or value > 3999:
        return str(value)
    result = ""
    for amount, numeral in _ROMAN:
        while value >= amount:
            result += numeral
            value -= amount
    return result
