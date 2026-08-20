"""Word equations (OMML) as LaTeX.

Word does not store an equation as text: it stores an ``m:oMath`` tree, and
``python-docx`` — which reads ``w:t`` runs — walks straight past it. A parser
built on it therefore drops every formula of a document silently, and an
inline equation takes the sentence around it with it ("Дисперсия равна
{formula} для выборки" arrives as "Дисперсия равна для выборки").

Docling, which this project already depends on, ships a full OMML→LaTeX
converter (fractions, radicals, n-ary operators with their limits, matrices,
accents, delimiters, function names). Reusing it beats writing a second one:
the interesting part here is *where* the equations belong — inline ones stay
inside their sentence as ``$…$``, a paragraph that is nothing but an equation
becomes a formula block — not how ``m:nary`` maps to ``\\sum``.

If that converter is ever unavailable, the equation still reaches the index as
the plain text of its symbols rather than disappearing.
"""

import logging
from typing import Any

from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

_OMATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"
_OMATH_PARA = "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara"
_MATH_TEXT = "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"

_converter: Any = None
_converter_loaded = False


def _load_converter() -> Any:
    global _converter, _converter_loaded
    if not _converter_loaded:
        _converter_loaded = True
        try:
            from docling.backend.docx.latex.omml import oMath2Latex

            _converter = oMath2Latex
        except Exception:  # pragma: no cover - depends on the Docling build
            logger.info("Docling's OMML converter is unavailable; equations kept as plain text.")
            _converter = None
    return _converter


def is_math(node: Any) -> bool:
    return node.tag in (_OMATH, _OMATH_PARA)


def outermost_math(element: Any) -> list[Any]:
    """Equations of an element, without the ``m:oMath`` nested in a wrapper."""
    return [node for node in element.iter() if is_math(node) and not _has_math_ancestor(node)]


def omml_to_latex(node: Any) -> str:
    """LaTeX for one ``m:oMath`` / ``m:oMathPara`` element."""
    converter = _load_converter()
    if converter is not None:
        try:
            if node.tag == _OMATH_PARA:
                parts = [str(converter(child)).strip() for child in node.iter(_OMATH)]
                latex = " ".join(part for part in parts if part)
            else:
                latex = str(converter(node)).strip()
            if latex:
                return latex
        except Exception:  # pragma: no cover - malformed or exotic equation
            logger.debug("Could not convert an equation to LaTeX.", exc_info=True)
    return math_plain_text(node)


def math_plain_text(node: Any) -> str:
    """The symbols of an equation, in order — the fallback that loses nothing."""
    return "".join(part.text or "" for part in node.iter(_MATH_TEXT)).strip()


def display_equation(paragraph_element: Any) -> str:
    """LaTeX of a paragraph that is *only* an equation, else an empty string.

    A display equation is its own block: it is the statement, not a phrase
    inside one, and keeping it separate is what lets a chunk cite "формула
    (3.14)" rather than a sentence that happens to contain it.
    """
    equations = [node for node in paragraph_element.iter() if is_math(node)]
    if not equations:
        return ""
    # ``oMathPara`` wraps its own ``oMath`` children; count only the outermost.
    outermost = [node for node in equations if not _has_math_ancestor(node)]
    if _paragraph_prose(paragraph_element):
        return ""
    return " ".join(filter(None, (omml_to_latex(node) for node in outermost)))


def _has_math_ancestor(node: Any) -> bool:
    parent = node.getparent()
    while parent is not None:
        if is_math(parent):
            return True
        parent = parent.getparent()
    return False


def _paragraph_prose(paragraph_element: Any) -> str:
    """Text of the paragraph outside its equations."""
    inside = {node for math in paragraph_element.iter() if is_math(math) for node in math.iter()}
    return "".join(
        node.text or "" for node in paragraph_element.iter(qn("w:t")) if node not in inside
    ).strip()
