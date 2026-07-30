from pathlib import Path

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class TXTParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8-sig")

        return Document(
            file_name=path.name,
            file_type="txt",
            blocks=[
                Block(
                    id="block-1",
                    text=text,
                    type="plain_text",
                    metadata={
                        "source_file": path.name,
                        "block_type": "plain_text",
                    },
                )
            ],
        )
