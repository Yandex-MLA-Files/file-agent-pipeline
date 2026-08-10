from pathlib import Path

from file_agent.agent import tools as tools_module
from file_agent.agent.sandbox import SandboxResult
from file_agent.agent.tools import (
    build_default_tools,
    calculate,
    run_python_on_spreadsheet,
    search_documents,
)
from file_agent.chunking import Chunk
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


def test_calculate_evaluates_arithmetic():
    result = calculate("(1200 - 950) / 950 * 100")

    assert result.content == str((1200 - 950) / 950 * 100)


def test_calculate_rejects_non_arithmetic_expressions():
    result = calculate("__import__('os').system('echo pwned')")

    assert result.content.startswith("Error:")


def test_calculate_rejects_division_by_zero():
    result = calculate("1 / 0")

    assert result.content.startswith("Error:")


def test_run_python_on_spreadsheet_rejects_unknown_file_name():
    result = run_python_on_spreadsheet({"known.xlsx": Path("known.xlsx")}, "missing.xlsx", "pass")

    assert "unknown file_name" in result.content
    assert "known.xlsx" in result.content


def test_run_python_on_spreadsheet_reports_timeout(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="", exit_code=1, timed_out=True, truncated=False
        ),
    )
    result = run_python_on_spreadsheet({"a.xlsx": Path("a.xlsx")}, "a.xlsx", "while True: pass")

    assert "timed out" in result.content


def test_run_python_on_spreadsheet_reports_exception_output(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="Traceback: KeyError", exit_code=1, timed_out=False, truncated=False
        ),
    )
    result = run_python_on_spreadsheet({"a.xlsx": Path("a.xlsx")}, "a.xlsx", "raise KeyError")

    assert "KeyError" in result.content


def test_run_python_on_spreadsheet_returns_stdout_on_success(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="42\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    result = run_python_on_spreadsheet({"a.xlsx": Path("a.xlsx")}, "a.xlsx", "print(42)")

    assert result.content == "42"


def test_build_default_tools_always_includes_search_and_calculate():
    tools = build_default_tools(FakeRetriever([]))

    assert {tool.name for tool in tools} == {"search_documents", "calculate"}


def test_build_default_tools_adds_spreadsheet_tool_only_for_xlsx_documents():
    without_xlsx = build_default_tools(
        FakeRetriever([]), document_paths={"notes.txt": Path("notes.txt")}
    )
    with_xlsx = build_default_tools(
        FakeRetriever([]), document_paths={"data.xlsx": Path("data.xlsx")}
    )

    assert "run_python_on_spreadsheet" not in {tool.name for tool in without_xlsx}
    assert "run_python_on_spreadsheet" in {tool.name for tool in with_xlsx}


def test_build_default_tools_search_schema_uses_the_given_default_top_k():
    tools = build_default_tools(FakeRetriever([]), default_top_k=3)

    search_tool = next(tool for tool in tools if tool.name == "search_documents")
    assert search_tool.parameters["properties"]["top_k"]["default"] == 3
