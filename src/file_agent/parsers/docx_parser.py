from pathlib import Path

from docx import Document as load_docx_document
from docx.table import Table
from docx.text.paragraph import Paragraph

from file_agent.document import Block, Document
from file_agent.parsers.base import BaseParser


class DOCXParser(BaseParser):
    def parse(self, file_path: Path) -> Document:
        path = Path(file_path)
        docx_document = load_docx_document(path)
        blocks: list[Block] = []
        paragraph_number = 0
        table_number = 0

        for content in docx_document.iter_inner_content():
            if isinstance(content, Paragraph):
                paragraph_number += 1
                text = content.text.strip()
                if not text:
                    continue

                blocks.append(
                    Block(
                        id=f"paragraph-{paragraph_number}",
                        text=text,
                        type="docx_paragraph",
                        metadata={
                            "source_file": path.name,
                            "paragraph_number": paragraph_number,
                        },
                    )
                )
                continue

            if isinstance(content, Table):
                table_number += 1
                text = _extract_table_text(content)
                if not text:
                    continue

                blocks.append(
                    Block(
                        id=f"table-{table_number}",
                        text=text,
                        type="docx_table",
                        metadata={
                            "source_file": path.name,
                            "table_number": table_number,
                            "rows_count": len(content.rows),
                            "columns_count": len(content.columns),
                        },
                    )
                )

        return Document(
            file_name=path.name,
            file_type="docx",
            blocks=blocks,
        )


def _extract_table_text(table: Table) -> str:
    rows: list[str] = []

    for row in table.rows:
        values = [cell.text.strip() for cell in row.cells]
        if any(values):
            rows.append("\t".join(values))

    return "\n".join(rows)
