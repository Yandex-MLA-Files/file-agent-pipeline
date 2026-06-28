from pathlib import Path

from file_agent.document import Document
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.md_parser import MarkdownParser
from file_agent.parsers.pdf_parser import PDFParser
from file_agent.parsers.pptx_parser import PPTXParser
from file_agent.parsers.xlsx_parser import XLSXParser


def parse_file(file_path: str | Path) -> Document:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".md":
        return MarkdownParser().parse(path)

    if suffix == ".pdf":
        return PDFParser().parse(path)

    if suffix in {".html", ".htm"}:
        return HTMLParser().parse(path)

    if suffix == ".xlsx":
        return XLSXParser().parse(path)

    if suffix == ".pptx":
        return PPTXParser().parse(path)

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")

