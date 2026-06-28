from pathlib import Path

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class MarkdownParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        return Document(
            file_name=path.name,
            file_type="md",
            blocks=[
                Block(
                    id="block-1",
                    text=text,
                    type="markdown",
                    metadata={"block_type": "markdown"},
                )
            ],
        )
