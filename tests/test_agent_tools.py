import pytest

from file_agent.agent.passages import PassageRegistry
from file_agent.agent.tools import (
    MAX_READ_CHARS,
    MAX_SEARCH_POOL,
    MAX_SEARCH_TOP_K,
    MIN_SEARCH_POOL,
    ToolError,
    build_default_tools,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult] | None = None, by_query: dict | None = None):
        self.results = results or []
        self.by_query = by_query or {}
        self.search_calls: list[tuple[str, int]] = []

    def index(self, chunks):
        pass

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return list(self.by_query.get(query, self.results))[:top_k]

    def clear(self):
        pass


def make_result(chunk_id: str, text: str, metadata: dict | None = None) -> SearchResult:
    base_metadata = {"source_file": "report.pdf"}
    base_metadata.update(metadata or {})
    return SearchResult(chunk=Chunk(id=chunk_id, text=text, metadata=base_metadata), score=0.9)


def heading(block_id: str, text: str, level: int = 1, page: int | None = None) -> Block:
    return Block(
        id=block_id,
        text=text,
        type="heading",
        metadata={"hierarchy_level": level, "source_file": "report.pdf"},
        block_type=BlockType.HEADING,
        page_number=page,
    )


def paragraph(block_id: str, text: str, page: int | None = None) -> Block:
    return Block(
        id=block_id,
        text=text,
        type="text",
        metadata={"source_file": "report.pdf", "dataset_doc_id": "doc-1"},
        block_type=BlockType.TEXT,
        page_number=page,
    )


def make_document() -> Document:
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            heading("h1", "Introduction", page=1),
            paragraph("p1", "The introduction text.", page=1),
            heading("h2", "Results", level=1, page=2),
            paragraph("p2", "Revenue grew by 10% to 96 083 million roubles.", page=2),
            heading("h3", "Details", level=2, page=3),
            paragraph("p3", "Detailed revenue table for Q1.", page=3),
            heading("h4", "Conclusion", level=1, page=4),
            paragraph("p4", "Closing remarks.", page=4),
        ],
    )
    document.build_table_of_contents()
    return document


def make_sheet_document() -> Document:
    inventory = (
        "id\tproduct\tstock\tprice\tcategory\n"
        "1\tLamp\t5\t10.5\tHome\n2\tChair\t20\t40\tHome\n3\tBall\t7\t3\tToys"
    )
    notes = "note\nData from 2016; see chessgoals.com for current numbers"
    return Document(
        file_name="inventory-data.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text=inventory,
                type="xlsx_sheet",
                metadata={
                    "source_file": "inventory-data.xlsx",
                    "sheet_name": "Inventory",
                    "max_row": 4,
                    "max_column": 5,
                },
            ),
            Block(
                id="sheet-2",
                text=notes,
                type="xlsx_sheet",
                metadata={
                    "source_file": "inventory-data.xlsx",
                    "sheet_name": "Notes",
                    "max_row": 2,
                    "max_column": 1,
                },
            ),
        ],
    )


def get_tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_build_default_tools_exposes_expected_names():
    tools = build_default_tools(FakeRetriever(), [make_document()])

    assert [tool.name for tool in tools] == [
        "search_documents",
        "find_text",
        "list_documents",
        "read_section",
        "read_pages",
        "read_document",
        "query_table",
        "calculate",
    ]
    for tool in tools:
        assert tool.name in tool.describe()
        assert tool.description


def test_search_documents_labels_passages_and_registers_them():
    results = [
        make_result("c1", "Chunk one", {"section": "Results", "page_numbers": [3, 4]}),
        make_result("c2", "Chunk two", {"page_number": 7}),
    ]
    registry = PassageRegistry()
    retriever = FakeRetriever(results)
    tool = get_tool(build_default_tools(retriever, [], registry), "search_documents")

    result = tool.run(query="revenue", top_k=2)

    assert retriever.search_calls == [("revenue", MIN_SEARCH_POOL)]
    assert [source.chunk.id for source in result.sources] == ["c1", "c2"]
    assert "[P1 | file=report.pdf | section=Results | pages=3, 4]" in result.output
    assert "[P2 | file=report.pdf | pages=7]" in result.output
    assert "Chunk one" in result.output
    assert [passage.id for passage in registry.all()] == ["P1", "P2"]


def test_search_documents_fuses_alternative_formulations():
    shared = make_result("c1", "Shared chunk")
    only_main = make_result("c2", "Main only")
    only_alt = make_result("c3", "Alt only")
    retriever = FakeRetriever(
        by_query={"main": [only_main, shared], "alt": [shared, only_alt]},
    )
    tool = get_tool(build_default_tools(retriever, []), "search_documents")

    result = tool.run(query="main", queries=["alt", "main"], top_k=3)

    # Both formulations ran once; the chunk found by both ranks first.
    assert [call[0] for call in retriever.search_calls] == ["main", "alt"]
    assert [source.chunk.id for source in result.sources] == ["c1", "c2", "c3"]
    assert "['main', 'alt']" in result.output


def test_search_documents_can_be_restricted_to_one_document():
    results = [
        make_result("a1", "From A", {"source_file": "a.pdf"}),
        make_result("b1", "From B", {"source_file": "b.pdf"}),
        make_result("b2", "From B too", {"source_file": "b.pdf"}),
    ]
    documents = [
        Document(file_name="a.pdf", file_type="pdf", blocks=[]),
        Document(file_name="b.pdf", file_type="pdf", blocks=[]),
    ]
    retriever = FakeRetriever(results)
    tool = get_tool(build_default_tools(retriever, documents), "search_documents")

    result = tool.run(query="q", file_name="b.pdf", top_k=5)

    assert {source.chunk.metadata["source_file"] for source in result.sources} == {"b.pdf"}
    assert retriever.search_calls[0][1] == min(MAX_SEARCH_POOL, max(MIN_SEARCH_POOL, 5 * 4))
    with pytest.raises(ToolError, match="not found"):
        tool.run(query="q", file_name="missing.pdf")


def test_search_documents_prefers_parent_context_and_deduplicates_it():
    shared_parent = "The whole section text."
    results = [
        make_result("c1", "First half", {"context": shared_parent}),
        make_result("c2", "Second half", {"context": shared_parent}),
    ]
    tool = get_tool(build_default_tools(FakeRetriever(results), []), "search_documents")

    result = tool.run(query="section")

    assert result.output.count(shared_parent) == 1
    assert "First half" not in result.output
    assert len(result.sources) == 1


def test_search_documents_clamps_top_k_and_validates_arguments():
    retriever = FakeRetriever([make_result("c1", "text")])
    tool = get_tool(build_default_tools(retriever, []), "search_documents")

    tool.run(query="q", top_k=999)
    assert retriever.search_calls == [("q", min(MAX_SEARCH_POOL, MAX_SEARCH_TOP_K * 2))]

    with pytest.raises(ToolError, match="query"):
        tool.run(query="   ")
    with pytest.raises(ToolError, match="top_k"):
        tool.run(query="q", top_k="many")
    with pytest.raises(ToolError, match="unknown argument"):
        tool.run(query="q", limit=3)


def test_search_documents_reports_empty_results():
    tool = get_tool(build_default_tools(FakeRetriever([]), []), "search_documents")

    result = tool.run(query="nothing")

    assert result.sources == []
    assert "No matching passages" in result.output


def test_find_text_returns_snippets_with_location():
    registry = PassageRegistry()
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()], registry), "find_text")

    result = tool.run(pattern="96 083")

    assert "1 match(es)" in result.output
    assert "[P1 | file=report.pdf | section=Results | pages=2]" in result.output
    assert "96 083 million" in result.output
    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "doc-1"


def test_find_text_is_case_insensitive_and_tolerates_yo():
    document = Document(
        file_name="notes.txt",
        file_type="txt",
        blocks=[paragraph("p1", "Учёт ведётся ежеквартально. Отчет сдан.")],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [document]), "find_text")

    assert "Учёт" in tool.run(pattern="учет").output
    assert "Отчет" in tool.run(pattern="ОТЧЁТ").output
    assert "No text matching" in tool.run(pattern="баланс").output


def test_find_text_returns_whole_spreadsheet_rows_with_headers():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_sheet_document()]), "find_text")

    result = tool.run(pattern="Chair")

    assert "id\tproduct\tstock\tprice\tcategory" in result.output
    assert "2\tChair\t20\t40\tHome" in result.output
    assert "sheet=Inventory" in result.output


def test_find_text_accepts_regular_expressions_and_limits_hits():
    document = Document(
        file_name="log.txt",
        file_type="txt",
        blocks=[paragraph(f"p{i}", f"Error code E{i:03d} reported.") for i in range(30)],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [document]), "find_text")

    result = tool.run(pattern=r"E\d{3}", max_hits=3)

    assert "30 match(es)" in result.output
    assert "showing 3 snippet(s)" in result.output
    assert len(result.sources) == 3


def test_list_documents_shows_structure_sheets_and_preview():
    tool = get_tool(
        build_default_tools(FakeRetriever(), [make_document(), make_sheet_document()]),
        "list_documents",
    )

    result = tool.run()

    assert "report.pdf (pdf; 4 page(s), 8 block(s))" in result.output
    assert "- Introduction (p. 1)" in result.output
    assert "  - Details (p. 3)" in result.output
    assert "inventory-data.xlsx (xlsx; 2 sheet(s))" in result.output
    assert "Inventory: 4 rows x 5 columns; first row: id | product | stock | price | category" in (
        result.output
    )
    assert "Begins with: The introduction text." not in result.output  # too short for a preview
    assert len(result.sources) == 2  # one overview passage per document


def test_list_documents_without_documents():
    tool = get_tool(build_default_tools(FakeRetriever(), []), "list_documents")

    assert tool.run().output == "No documents are loaded."


def test_read_section_returns_heading_with_nested_subsections():
    registry = PassageRegistry()
    tool = get_tool(
        build_default_tools(FakeRetriever(), [make_document()], registry), "read_section"
    )

    result = tool.run(file_name="report.pdf", section="Results")

    assert "# Results" in result.output
    assert "Revenue grew by 10%" in result.output
    assert "## Details" in result.output
    assert "Detailed revenue table" in result.output
    assert "Conclusion" not in result.output.split("]", 1)[1]
    assert "section=Results" in result.output
    assert "pages=2, 3" in result.output
    assert result.sources[0].chunk.metadata["tool"] == "read_section"


def test_read_section_matches_heading_and_file_loosely():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "read_section")

    assert "Closing remarks" in tool.run(file_name="REPORT", section="conclusion").output
    assert "Closing remarks" in tool.run(file_name="report", section="4 Conclusion").output


def test_read_section_serves_long_sections_in_parts():
    long_text = " ".join(f"sentence {i}." for i in range(3000))
    document = Document(
        file_name="long.md",
        file_type="md",
        blocks=[heading("h1", "Body"), paragraph("p1", long_text)],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [document]), "read_section")

    first = tool.run(file_name="long.md", section="Body")
    assert "part 1 of" in first.output
    assert "Continue with part=2" in first.output
    assert len(first.sources[0].chunk.text) <= MAX_READ_CHARS

    second = tool.run(file_name="long.md", section="Body", part=2)
    assert "part 2 of" in second.output
    assert second.sources[0].chunk.text != first.sources[0].chunk.text
    with pytest.raises(ToolError, match="part"):
        tool.run(file_name="long.md", section="Body", part=99)


def test_read_section_errors_are_actionable():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "read_section")

    with pytest.raises(ToolError, match="Available documents: report.pdf"):
        tool.run(file_name="missing.pdf", section="Results")
    with pytest.raises(ToolError, match="Sections: Introduction; Results"):
        tool.run(file_name="report.pdf", section="Appendix")
    with pytest.raises(ToolError, match="both"):
        tool.run(file_name="report.pdf")


def test_read_pages_returns_one_passage_per_page():
    registry = PassageRegistry()
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()], registry), "read_pages")

    result = tool.run(file_name="report.pdf", pages="2-3")

    assert "[P1 | file=report.pdf | pages=2]" in result.output
    assert "[P2 | file=report.pdf | pages=3]" in result.output
    assert "Revenue grew" in result.output
    assert "Closing remarks" not in result.output
    with pytest.raises(ToolError, match="No content on pages"):
        tool.run(file_name="report.pdf", pages="40")
    with pytest.raises(ToolError, match="pages"):
        tool.run(file_name="report.pdf", pages="abc")


def test_read_pages_reads_sheets_by_number():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_sheet_document()]), "read_pages")

    result = tool.run(file_name="inventory-data.xlsx", pages=2)

    assert "chessgoals.com" in result.output
    assert "Lamp" not in result.output


def test_read_document_serves_the_whole_text_in_parts():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "read_document")

    result = tool.run(file_name="report.pdf")

    assert "part 1 of 1" in result.output
    assert "# Introduction" in result.output
    assert "Closing remarks" in result.output
    assert len(result.sources) == 1


def test_query_table_previews_and_computes_over_sheets():
    registry = PassageRegistry()
    tool = get_tool(
        build_default_tools(FakeRetriever(), [make_sheet_document()], registry), "query_table"
    )

    preview = tool.run(file_name="inventory-data.xlsx")
    assert "Tables in 'inventory-data.xlsx': 'Inventory', 'Notes'" in preview.output
    assert "3 rows x 5 columns: id, product, stock, price, category" in preview.output

    computed = tool.run(
        file_name="inventory-data.xlsx",
        code=(
            "total = (df['stock'] * df['price']).sum()\nprint(round(total, 2))\n"
            "df.groupby('category')['stock'].sum()"
        ),
    )
    assert "873.5" in computed.output  # 5*10.5 + 20*40 + 7*3
    assert "Home" in computed.output and "25" in computed.output
    assert computed.sources[0].chunk.metadata["sheet_name"] == "Inventory"
    assert "sheet=Inventory" in computed.output

    other = tool.run(file_name="inventory-data.xlsx", sheet="Notes", code="len(df)")
    assert "=>\n1" in other.output


def test_query_table_reports_code_errors_and_missing_tables():
    tool = get_tool(
        build_default_tools(FakeRetriever(), [make_sheet_document(), make_document()]),
        "query_table",
    )

    with pytest.raises(ToolError, match="KeyError"):
        tool.run(file_name="inventory-data.xlsx", code="df['missing'].sum()")
    with pytest.raises(ToolError, match="Imports are not allowed"):
        tool.run(file_name="inventory-data.xlsx", code="import os")
    with pytest.raises(ToolError, match="No sheet or table named"):
        tool.run(file_name="inventory-data.xlsx", sheet="Orders", code="len(df)")
    with pytest.raises(ToolError, match="no sheets or tables"):
        tool.run(file_name="report.pdf", code="len(df)")


def test_query_table_parses_markdown_tables_inside_documents():
    table = Block(
        id="t1",
        text="| name | score |\n|---|---|\n| A | 10 |\n| B | 30 |",
        type="table",
        metadata={"source_file": "paper.pdf"},
        block_type=BlockType.TABLE,
        page_number=5,
    )
    document = Document(file_name="paper.pdf", file_type="pdf", blocks=[table])
    tool = get_tool(build_default_tools(FakeRetriever(), [document]), "query_table")

    result = tool.run(file_name="paper.pdf", code="df['score'].mean()")

    assert "20.0" in result.output
    assert result.sources[0].chunk.metadata["table"] == "table1"
    assert result.sources[0].chunk.metadata["page_number"] == 5


def test_calculate_evaluates_expressions():
    tool = get_tool(build_default_tools(FakeRetriever(), []), "calculate")

    result = tool.run(expression="round(3.81e6 / 1.52e6, 2)")

    assert "= 2.51" in result.output
    assert result.sources[0].chunk.metadata["tool"] == "calculate"
    with pytest.raises(ToolError, match="calculate failed"):
        tool.run(expression="open('x')")
