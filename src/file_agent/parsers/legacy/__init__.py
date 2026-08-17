"""Original single-block parsers kept as a fallback implementation.

Selected with ``PARSER_PROFILE=legacy`` (see :mod:`file_agent.pipeline`).
They emit one flat text block per slide/sheet/file without headings, tables or
figures; the structured parsers in :mod:`file_agent.parsers` supersede them.
"""

from file_agent.parsers.legacy.html_parser import HTMLParser
from file_agent.parsers.legacy.md_parser import MarkdownParser
from file_agent.parsers.legacy.pptx_parser import PPTXParser
from file_agent.parsers.legacy.txt_parser import TXTParser
from file_agent.parsers.legacy.xlsx_parser import XLSXParser

__all__ = ["HTMLParser", "MarkdownParser", "PPTXParser", "TXTParser", "XLSXParser"]
