from pathlib import Path

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.base import BaseParser


class ImageParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        return Document(
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            blocks=[
                Block(
                    id="block-1",
                    text=f"[Image file: {path.name}]",
                    type="figure",
                    block_type=BlockType.FIGURE,
                    page_number=1,
                    metadata={"block_type": "figure"},
                )
            ],
        )
