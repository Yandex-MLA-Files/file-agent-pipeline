from pathlib import Path

import fitz

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        blocks: list[Block] = []

        with fitz.open(path) as pdf_document:
            for page_index, page in enumerate(pdf_document, start=1):
                blocks.append(
                    Block(
                        id=f"page-{page_index}",
                        text=page.get_text(),
                        type="pdf_page",
                        metadata={
                            "page_number": page_index,
                            "source_file": path.name,
                        },
                    )
                )

        return Document(
            file_name=path.name,
            file_type="pdf",
            blocks=blocks,
        )
