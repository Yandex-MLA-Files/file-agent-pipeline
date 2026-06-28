from pathlib import Path

from file_agent.document import Document
from file_agent.parsers.md_parser import MarkdownParser


def parse_file(file_path: str | Path) -> Document:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".md":
        return MarkdownParser().parse(path)

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")
