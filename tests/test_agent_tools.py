import json
from pathlib import Path

from file_agent.agent import tools as tools_module
from file_agent.agent.sandbox import SandboxResult
from file_agent.agent.tools import (
    build_default_tools,
    list_documents,
    run_python,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, Document
from file_agent.retrieval import SearchResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return self.results[:top_k]

    def index(self, chunks):
        raise AssertionError("unused")

    def clear(self):
        raise AssertionError("unused")


def make_result(text: str, score: float) -> SearchResult:
    return SearchResult(chunk=Chunk(id=text, text=text, metadata={}), score=score)


def test_search_documents_returns_formatted_context_and_sources():
    results = [make_result("First passage", 1.0), make_result("Second passage", 0.5)]
    retriever = FakeRetriever(results)

    result = search_documents(retriever, "some query", top_k=2)

    assert "First passage" in result.content
    assert "Second passage" in result.content
    assert result.sources == results
    assert retriever.search_calls == [("some query", 2)]


def test_search_documents_reports_no_matches_without_crashing():
    result = search_documents(FakeRetriever([]), "nothing here")

    assert result.content == "No matching passages found."
    assert result.sources == []


def test_run_python_reports_timeout(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="", exit_code=1, timed_out=True, truncated=False
        ),
    )
    result = run_python({"a.xlsx": Path("a.xlsx")}, "while True: pass")

    assert "timed out" in result.content


def test_run_python_reports_exception_output(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="Traceback: KeyError", exit_code=1, timed_out=False, truncated=False
        ),
    )
    result = run_python({"a.xlsx": Path("a.xlsx")}, "raise KeyError")

    assert "KeyError" in result.content


def test_run_python_returns_stdout_on_success(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="42\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    result = run_python({"a.xlsx": Path("a.xlsx")}, "print(42)")

    assert result.content == "42"


def test_run_python_attaches_evidence_source_on_successful_output(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="7366.0\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    result = run_python({}, "print(2250 + 2170 + 1536 + 1410)")

    assert len(result.sources) == 1
    source = result.sources[0]
    assert "2250 + 2170 + 1536 + 1410" in source.chunk.text
    assert "7366.0" in source.chunk.text
    assert source.chunk.metadata["dataset_doc_id"] == "run_python"
    assert source.score == 1.0


def test_run_python_attaches_no_evidence_when_there_is_no_output(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    result = run_python({}, "x = 1")

    assert result.sources == []


def test_run_python_attaches_no_evidence_on_timeout(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="", exit_code=1, timed_out=True, truncated=False
        ),
    )
    result = run_python({}, "while True: pass")

    assert result.sources == []


def test_run_python_attaches_the_error_as_evidence(monkeypatch):
    """A failed run is the only thing an answer like "there is no such sheet"
    can rest on; without evidence RagasJudge scores that answer as ungrounded."""
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="",
            stderr="ValueError: Worksheet named 'Comparison' not found",
            exit_code=1,
            timed_out=False,
            truncated=False,
        ),
    )
    result = run_python({}, "pd.read_excel('/data/a.xlsx', sheet_name='Comparison')")

    assert len(result.sources) == 1
    assert "Worksheet named 'Comparison' not found" in result.sources[0].chunk.text
    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "run_python"


def test_run_python_attaches_no_evidence_when_a_failure_says_nothing(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="", exit_code=1, timed_out=False, truncated=False
        ),
    )
    result = run_python({}, "raise SystemExit(1)")

    assert result.sources == []


def test_run_python_works_without_any_documents(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="7\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    result = run_python({}, "print(3 + 4)")

    assert result.content == "7"


def test_list_documents_summarizes_pages_sheets_and_headings():
    pdf_document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="Intro", type="heading", page_number=1)],
        metadata={"total_pages": 3, "table_of_contents": [{"title": "Intro"}]},
    )
    xlsx_document = Document(
        file_name="data.xlsx",
        file_type=".xlsx",
        blocks=[
            Block(id="b2", text="1\t2", type="xlsx_sheet", metadata={"sheet_name": "Sheet1"}),
            Block(id="b3", text="3\t4", type="xlsx_sheet", metadata={"sheet_name": "Sheet2"}),
        ],
    )

    result = list_documents([pdf_document, xlsx_document])
    documents = json.loads(result.content)["documents"]

    assert documents[0]["file_name"] == "report.pdf"
    assert documents[0]["total_pages"] == 3
    assert documents[0]["headings"] == 1
    assert documents[1]["file_name"] == "data.xlsx"
    assert documents[1]["sheets"] == ["Sheet1", "Sheet2"]


def test_list_documents_attaches_its_output_as_evidence():
    """Sheet/page/slide questions are answered from this output alone, so it
    has to reach the exported contexts - otherwise a correct answer scores 0."""
    xlsx_document = Document(
        file_name="data.xlsx",
        file_type=".xlsx",
        blocks=[Block(id="b1", text="1\t2", type="xlsx_sheet", metadata={"sheet_name": "Sheet1"})],
    )

    result = list_documents([xlsx_document])

    assert len(result.sources) == 1
    source = result.sources[0]
    assert "data.xlsx" in source.chunk.text
    assert "Sheet1" in source.chunk.text
    assert source.chunk.metadata["dataset_doc_id"] == "list_documents"
    assert source.score == 1.0


def test_list_documents_handles_no_documents():
    result = list_documents([])

    assert result.content == '{"documents": []}'
    assert result.sources == []


def test_build_default_tools_always_includes_all_three_tools():
    tools = build_default_tools(FakeRetriever([]))

    assert {tool.name for tool in tools} == {"search_documents", "list_documents", "run_python"}


def test_build_default_tools_run_python_description_lists_available_files():
    tools = build_default_tools(
        FakeRetriever([]),
        document_paths={"data.xlsx": Path("data.xlsx"), "notes.txt": Path("notes.txt")},
    )

    run_python_tool = next(tool for tool in tools if tool.name == "run_python")
    assert "data.xlsx" in run_python_tool.description
    assert "notes.txt" in run_python_tool.description


def test_build_default_tools_search_schema_uses_the_given_default_top_k():
    tools = build_default_tools(FakeRetriever([]), default_top_k=3)

    search_tool = next(tool for tool in tools if tool.name == "search_documents")
    assert search_tool.parameters["properties"]["top_k"]["default"] == 3
