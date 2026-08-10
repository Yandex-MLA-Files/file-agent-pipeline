import pytest

from file_agent.agent.tools import (
    MAX_SEARCH_TOP_K,
    MAX_SECTION_CHARS,
    ToolError,
    build_default_tools,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results or []
        self.search_calls: list[tuple[str, int]] = []

    def index(self, chunks):
        pass

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return self.results[:top_k]

    def clear(self):
        pass


def make_result(chunk_id: str, text: str, metadata: dict | None = None) -> SearchResult:
    base_metadata = {"source_file": "report.pdf"}
    base_metadata.update(metadata or {})
    return SearchResult(chunk=Chunk(id=chunk_id, text=text, metadata=base_metadata), score=0.9)


def heading(block_id: str, text: str, level: int = 1) -> Block:
    return Block(
        id=block_id,
        text=text,
        type="heading",
        metadata={"hierarchy_level": level},
        block_type=BlockType.HEADING,
    )


def paragraph(block_id: str, text: str) -> Block:
    return Block(id=block_id, text=text, type="text", block_type=BlockType.TEXT)


def make_document() -> Document:
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            heading("h1", "Introduction"),
            paragraph("p1", "The introduction text."),
            heading("h2", "Results", level=1),
            paragraph("p2", "Revenue grew by 10%."),
            heading("h3", "Details", level=2),
            paragraph("p3", "Detailed revenue table."),
            heading("h4", "Conclusion", level=1),
            paragraph("p4", "Closing remarks."),
        ],
    )
    document.build_table_of_contents()
    return document


def get_tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_build_default_tools_exposes_expected_names():
    tools = build_default_tools(FakeRetriever(), [make_document()])

    assert [tool.name for tool in tools] == [
        "search_documents",
        "list_documents",
        "read_section",
    ]
    for tool in tools:
        assert tool.describe().startswith(f"- {tool.name}")


def test_search_documents_formats_passages_with_source_metadata():
    results = [
        make_result("c1", "Chunk one", {"section": "Results", "page_numbers": [3, 4]}),
        make_result("c2", "Chunk two", {"page_number": 7}),
    ]
    retriever = FakeRetriever(results)
    tool = get_tool(build_default_tools(retriever, []), "search_documents")

    result = tool.run(query="revenue", top_k=2)

    assert retriever.search_calls == [("revenue", 2)]
    assert result.sources == results
    assert "[Passage 1" in result.output
    assert "file=report.pdf" in result.output
    assert "section=Results" in result.output
    assert "pages=3, 4" in result.output
    assert "pages=7" in result.output
    assert "Chunk one" in result.output


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


def test_search_documents_clamps_top_k_and_validates_arguments():
    retriever = FakeRetriever([make_result("c1", "text")])
    tool = get_tool(build_default_tools(retriever, []), "search_documents")

    tool.run(query="q", top_k=999)
    assert retriever.search_calls == [("q", MAX_SEARCH_TOP_K)]

    with pytest.raises(ToolError, match="query"):
        tool.run(query="   ")
    with pytest.raises(ToolError, match="top_k"):
        tool.run(query="q", top_k="many")
    with pytest.raises(ToolError, match="unknown argument"):
        tool.run(query="q", limit=3)


def test_search_documents_reports_empty_results():
    tool = get_tool(build_default_tools(FakeRetriever([]), []), "search_documents")

    result = tool.run(query="nothing")

    assert "No matching passages" in result.output
    assert result.sources == []


def test_list_documents_shows_toc_and_counts():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "list_documents")

    result = tool.run()

    assert "report.pdf" in result.output
    assert "Table of contents" in result.output
    assert "Introduction" in result.output
    assert "Details" in result.output


def test_list_documents_without_documents_and_without_toc():
    empty_tool = get_tool(build_default_tools(FakeRetriever(), []), "list_documents")
    assert "No documents" in empty_tool.run().output

    plain = Document(
        file_name="notes.txt",
        file_type="txt",
        blocks=[paragraph("p1", "Just text.")],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [plain]), "list_documents")
    assert "No table of contents" in tool.run().output


def test_read_section_returns_heading_with_nested_subsections():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "read_section")

    result = tool.run(file_name="report.pdf", section="Results")

    assert "Results" in result.output
    assert "Revenue grew by 10%." in result.output
    # The nested level-2 subsection belongs to "Results".
    assert "Detailed revenue table." in result.output
    # The next level-1 section does not.
    assert "Closing remarks." not in result.output


def test_read_section_matches_heading_and_file_case_insensitively():
    tool = get_tool(build_default_tools(FakeRetriever(), [make_document()]), "read_section")

    result = tool.run(file_name="REPORT.pdf", section="conclu")

    assert "Closing remarks." in result.output


def test_read_section_truncates_long_sections():
    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            heading("h1", "Big"),
            paragraph("p1", "word " * 3000),
        ],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [document]), "read_section")

    result = tool.run(file_name="report.pdf", section="Big")

    assert len(result.output) < MAX_SECTION_CHARS + 200
    assert "Section truncated" in result.output


def test_read_section_errors_are_actionable():
    tools = build_default_tools(FakeRetriever(), [make_document()])
    tool = get_tool(tools, "read_section")

    with pytest.raises(ToolError, match="file_name"):
        tool.run(file_name="", section="Results")
    with pytest.raises(ToolError, match="Available documents: report.pdf"):
        tool.run(file_name="other.pdf", section="Results")
    with pytest.raises(ToolError, match="Sections: Introduction"):
        tool.run(file_name="report.pdf", section="Missing chapter")


def test_read_section_requires_detected_headings():
    plain = Document(
        file_name="notes.txt",
        file_type="txt",
        blocks=[paragraph("p1", "Just text.")],
    )
    tool = get_tool(build_default_tools(FakeRetriever(), [plain]), "read_section")

    with pytest.raises(ToolError, match="no detected sections"):
        tool.run(file_name="notes.txt", section="Anything")
