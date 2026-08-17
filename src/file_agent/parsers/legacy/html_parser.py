from pathlib import Path

from bs4 import BeautifulSoup

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class HTMLParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return Document(
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            blocks=[
                Block(
                    id="block-1",
                    text=text,
                    type="html_text",
                    metadata={"source_file": path.name},
                )
            ],
        )
