import json
from pathlib import Path

from PIL import Image

from file_agent.agent import tools as tools_module
from file_agent.agent.sandbox import SandboxResult
from file_agent.agent.tools import (
    ALL_TOOL_NAMES,
    MAX_PAGE_CHARS,
    build_default_tools,
    describe_image,
    list_documents,
    read_page,
    run_python,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import SearchResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def search(self, query: str, top_k: int = 5, source_file: str | None = None):
        self.search_calls.append((query, top_k, source_file))
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
    assert retriever.search_calls == [("some query", 2, None)]


def test_search_documents_reports_no_matches_without_crashing():
    result = search_documents(FakeRetriever([]), "nothing here")

    assert result.content == "No matching passages found."
    assert result.sources == []


def test_search_documents_passes_a_known_source_file_to_the_retriever():
    results = [make_result("First passage", 1.0)]
    retriever = FakeRetriever(results)
    documents = [Document(file_name="report.pdf", file_type="pdf", blocks=[])]

    result = search_documents(
        retriever, "some query", top_k=3, source_file="report.pdf", documents=documents
    )

    assert result.sources == results
    assert retriever.search_calls == [("some query", 3, "report.pdf")]


def test_search_documents_rejects_an_unknown_source_file():
    retriever = FakeRetriever([make_result("First passage", 1.0)])
    documents = [Document(file_name="report.pdf", file_type="pdf", blocks=[])]

    result = search_documents(
        retriever, "some query", source_file="missing.pdf", documents=documents
    )

    assert "no document named" in result.content
    assert "report.pdf" in result.content
    assert retriever.search_calls == []


def test_search_documents_skips_validation_without_a_known_documents_list():
    # documents=None (caller didn't pass it) - trust the retriever/source_file
    # rather than blocking a legitimate filter just because we can't validate it.
    results = [make_result("First passage", 1.0)]
    retriever = FakeRetriever(results)

    result = search_documents(retriever, "some query", source_file="report.pdf")

    assert result.sources == results
    assert retriever.search_calls == [("some query", 5, "report.pdf")]


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


def test_run_python_attributes_evidence_to_the_real_document_it_reads(monkeypatch):
    # Regression: run_python's evidence used to always carry the generic
    # "run_python" sentinel as dataset_doc_id, so an answer computed entirely
    # by reading a DOCX table/section (no search_documents call) couldn't be
    # matched against the question's real doc_ids by eval metrics - it looked
    # ungrounded even when it wasn't.
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="found it\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    document = Document(
        file_name="report.docx",
        file_type="docx",
        blocks=[
            Block(
                id="b1",
                text="...",
                type="text",
                metadata={"dataset_doc_id": "q0092/report.docx"},
            )
        ],
    )

    result = run_python(
        {"report.docx": Path("report.docx")},
        "from docx import Document\ndoc = Document('/data/report.docx')",
        documents=[document],
    )

    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "q0092/report.docx"


def test_run_python_falls_back_to_the_file_name_when_blocks_carry_no_dataset_doc_id(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    document = Document(
        file_name="report.docx",
        file_type="docx",
        blocks=[Block(id="b1", text="...", type="text", metadata={})],
    )

    result = run_python(
        {"report.docx": Path("report.docx")},
        "open('/data/report.docx')",
        documents=[document],
    )

    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "report.docx"


def test_run_python_uses_the_sentinel_when_code_references_several_documents(monkeypatch):
    # Attributing to one specific document would be a guess when the code
    # visibly reads more than one - the generic sentinel is the honest answer.
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        ),
    )
    documents = [
        Document(file_name="a.docx", file_type="docx", blocks=[]),
        Document(file_name="b.docx", file_type="docx", blocks=[]),
    ]

    result = run_python(
        {"a.docx": Path("a.docx"), "b.docx": Path("b.docx")},
        "open('/data/a.docx'); open('/data/b.docx')",
        documents=documents,
    )

    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "run_python"


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


def test_run_python_repeats_the_schema_hint_next_to_the_error(monkeypatch):
    """A KeyError on a guessed column is exactly the failure the schema hint
    exists to prevent - it needs to be visible right where the model is
    already looking (the traceback), not just once at the top of the
    conversation it may not re-read."""
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="",
            stderr="KeyError: 'Category'",
            exit_code=1,
            timed_out=False,
            truncated=False,
        ),
    )
    result = run_python({}, "df['Category']", schema_hint="\nSpreadsheet layout: id, product\n")

    assert "KeyError: 'Category'" in result.content
    assert "Spreadsheet layout: id, product" in result.content


def test_run_python_omits_the_schema_hint_when_none_is_given(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "run_sandboxed_code",
        lambda **kwargs: SandboxResult(
            stdout="", stderr="boom", exit_code=1, timed_out=False, truncated=False
        ),
    )
    result = run_python({}, "raise ValueError('boom')")

    assert result.content == "Error: code raised an exception:\nboom"


def test_run_python_mounts_accumulated_context_as_a_readonly_file(monkeypatch, tmp_path):
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["source_paths"] = dict(source_paths)
        context_path = source_paths["_context.json"]
        captured["context_contents"] = json.loads(context_path.read_text(encoding="utf-8"))
        assert context_path.is_file()
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    context = [
        SearchResult(chunk=Chunk(id="c1", text="Found passage", metadata={"page": 1}), score=0.9)
    ]
    result = run_python({}, "print('ok')", context=context)

    assert result.content == "ok"
    assert set(captured["source_paths"]) == {"_context.json"}
    assert captured["context_contents"] == {
        "previous_tool_results": [{"text": "Found passage", "metadata": {"page": 1}}]
    }
    # The mounted temp file must not leak past the call.
    assert not captured["source_paths"]["_context.json"].exists()


def test_run_python_deduplicates_context_passages(monkeypatch):
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["context"] = json.loads(source_paths["_context.json"].read_text(encoding="utf-8"))
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    context = [
        SearchResult(chunk=Chunk(id="c1", text="Same passage", metadata={}), score=0.9),
        SearchResult(chunk=Chunk(id="c2", text="Same passage", metadata={}), score=0.5),
    ]
    run_python({}, "print('ok')", context=context)

    assert len(captured["context"]["previous_tool_results"]) == 1


def test_run_python_has_no_context_file_when_nothing_accumulated(monkeypatch):
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["source_paths"] = dict(source_paths)
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    run_python({}, "print('ok')")

    assert captured["source_paths"] == {}


def test_full_document_text_joins_pages_in_order_and_skips_empty_blocks():
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(id="b1", text="Page two text", type="text", page_number=2),
            Block(id="b2", text="Page one text", type="text", page_number=1),
            Block(id="b3", text="", type="figure", block_type=BlockType.FIGURE, page_number=1),
            Block(id="b4", text="  ", type="text", page_number=3),  # whitespace-only
        ],
    )

    text = tools_module._full_document_text(document)

    assert text == "=== page 1 ===\nPage one text\n\n=== page 2 ===\nPage two text"


def test_run_python_mounts_full_document_text_for_pdf(monkeypatch, tmp_path):
    # The whole point: PDF has no structured reader in the sandbox, but its
    # already-extracted text should still be readable as a plain .txt file.
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["source_paths"] = dict(source_paths)
        text_path = source_paths["report.pdf.txt"]
        captured["text_contents"] = text_path.read_text(encoding="utf-8")
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="References [1] [2] [3]", type="text", page_number=9)],
    )

    result = run_python({}, "print('ok')", documents=[document])

    assert result.content == "ok"
    assert "References [1] [2] [3]" in captured["text_contents"]
    # The mounted temp file must not leak past the call, same as _context.json.
    assert not captured["source_paths"]["report.pdf.txt"].exists()


def test_run_python_skips_a_document_text_file_when_it_has_no_extractable_text():
    document = Document(file_name="empty.pdf", file_type=".pdf", blocks=[])

    paths = tools_module._write_document_text_files([document])

    assert paths == {}


def test_document_tables_numbers_tables_in_document_order_and_skips_non_tables():
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(id="b1", text="Intro paragraph", type="text", page_number=1),
            Block(
                id="b2",
                text="| Model | AUPRC |\n| KGLM | 0.506 |",
                type="table",
                block_type=BlockType.TABLE,
                page_number=2,
            ),
            Block(
                id="b3",
                text="| Dataset | Kappa |\n| FEVER | 0.854 |",
                type="table",
                block_type=BlockType.TABLE,
                page_number=5,
            ),
        ],
    )

    tables = tools_module._document_tables(document)

    assert [t["table_index"] for t in tables] == [1, 2]
    assert tables[0]["page_number"] == 2
    assert tables[0]["file_name"] == "report.pdf"
    assert "AUPRC" in tables[0]["markdown"]
    assert tables[1]["page_number"] == 5


def test_write_tables_file_returns_none_when_no_tables_exist():
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="Just text", type="text", page_number=1)],
    )

    assert tools_module._write_tables_file([document]) is None


def test_run_python_mounts_tables_json_when_tables_exist(monkeypatch):
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["source_paths"] = dict(source_paths)
        captured["tables_contents"] = json.loads(
            source_paths["_tables.json"].read_text(encoding="utf-8")
        )
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(
                id="b1",
                text="| Model | AUPRC |\n| KGLM | 0.506 |",
                type="table",
                block_type=BlockType.TABLE,
                page_number=2,
            )
        ],
    )

    result = run_python({}, "print('ok')", documents=[document])

    assert result.content == "ok"
    assert captured["tables_contents"] == {
        "tables": [
            {
                "file_name": "report.pdf",
                "page_number": 2,
                "table_index": 1,
                "markdown": "| Model | AUPRC |\n| KGLM | 0.506 |",
            }
        ]
    }
    # The mounted temp file must not leak past the call, same as _context.json.
    assert not captured["source_paths"]["_tables.json"].exists()


def test_run_python_has_no_tables_file_when_there_are_no_tables(monkeypatch):
    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["source_paths"] = dict(source_paths)
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)

    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="Just text", type="text", page_number=1)],
    )

    run_python({}, "print('ok')", documents=[document])

    assert "_tables.json" not in captured["source_paths"]


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
    assert [sheet["name"] for sheet in documents[1]["sheets"]] == ["Sheet1", "Sheet2"]


def test_list_documents_reports_spreadsheet_columns_and_row_counts():
    """Structure questions about spreadsheets ask for exactly this ("which
    columns", "how many rows"); the sheet name alone left the agent guessing."""
    xlsx_document = Document(
        file_name="sales.xlsx",
        file_type=".xlsx",
        blocks=[
            Block(
                id="b1",
                text="id\tproduct\tamount\n1\tpen\t10\n2\tcup\t20",
                type="xlsx_sheet",
                metadata={"sheet_name": "Sales", "max_row": 3, "max_column": 3},
            )
        ],
    )

    sheet = json.loads(list_documents([xlsx_document]).content)["documents"][0]["sheets"][0]

    assert sheet["name"] == "Sales"
    assert sheet["columns"] == ["id", "product", "amount"]
    assert sheet["data_rows"] == 2


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


def test_list_documents_reports_pages_with_undescribed_images():
    # Regression: an undescribed figure/image block has no text, so it's
    # invisible to search_documents and read_page's assembled text alike -
    # this is the only place its existence is surfaced to the agent at all.
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(id="b1", text="Intro", type="text", page_number=1),
            Block(id="b2", text="", type="figure", block_type=BlockType.FIGURE, page_number=3),
            Block(id="b3", text="", type="image", block_type=BlockType.IMAGE, page_number=3),
            Block(id="b4", text="", type="figure", block_type=BlockType.FIGURE, page_number=5),
            # Already described (eager VLM path) - not "undescribed", shouldn't be listed.
            Block(
                id="b5",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=9,
                vlm_description="A bar chart.",
            ),
        ],
    )

    entry = json.loads(list_documents([document]).content)["documents"][0]

    assert entry["pages_with_undescribed_images"] == [3, 5]


def test_list_documents_reports_no_undescribed_images_when_there_are_none():
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="Intro", type="text", page_number=1)],
    )

    entry = json.loads(list_documents([document]).content)["documents"][0]

    assert entry["pages_with_undescribed_images"] == []


def make_paged_document() -> Document:
    return Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(
                id="b1",
                text="Revenue table",
                type="text",
                page_number=7,
                metadata={"dataset_doc_id": "q0018/report.pdf"},
            ),
            Block(
                id="b2",
                text="Total 940123",
                type="table",
                page_number=7,
                metadata={"dataset_doc_id": "q0018/report.pdf"},
            ),
            Block(id="b3", text="Other page", type="text", page_number=8),
        ],
    )


def test_read_page_returns_the_whole_page_as_evidence():
    result = read_page([make_paged_document()], [{"file_name": "report.pdf", "page": 7}])

    assert "Revenue table" in result.content
    assert "Total 940123" in result.content
    assert "Other page" not in result.content
    assert len(result.sources) == 1
    # Navigation reaches real document content, so it keeps the real doc id
    # instead of a tool sentinel like run_python's.
    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "q0018/report.pdf"
    assert result.sources[0].chunk.metadata["page_number"] == 7


def test_read_page_reports_an_unknown_file_with_the_available_names():
    result = read_page([make_paged_document()], [{"file_name": "missing.pdf", "page": 1}])

    assert "no document named" in result.content
    assert "report.pdf" in result.content
    assert result.sources == []


def test_read_page_reports_an_out_of_range_page_with_the_real_range():
    result = read_page([make_paged_document()], [{"file_name": "report.pdf", "page": 99}])

    assert "no page 99" in result.content
    assert "1-8" in result.content
    assert result.sources == []


def test_read_page_notes_an_undescribed_figure_on_the_page():
    # Regression: the figure block contributes nothing to the assembled page
    # text (see _is_undescribed_figure), so without this note the agent has
    # no way to learn a figure is there at all once it's read the page.
    document = Document(
        file_name="report.pdf",
        file_type=".pdf",
        blocks=[
            Block(id="b1", text="Revenue table", type="text", page_number=7),
            Block(id="b2", text="", type="figure", block_type=BlockType.FIGURE, page_number=7),
        ],
    )

    result = read_page([document], [{"file_name": "report.pdf", "page": 7}])

    assert "Revenue table" in result.content
    assert "describe_image" in result.content
    assert "report.pdf" in result.content
    assert "page=7" in result.content


def test_read_page_says_nothing_about_figures_when_the_page_has_none():
    result = read_page([make_paged_document()], [{"file_name": "report.pdf", "page": 7}])

    assert "describe_image" not in result.content


def test_read_page_reports_a_figure_on_a_page_with_no_extractable_text():
    document = Document(
        file_name="deck.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="", type="figure", block_type=BlockType.FIGURE, page_number=4)],
    )

    result = read_page([document], [{"file_name": "deck.pdf", "page": 4}])

    assert "no extractable text" in result.content
    assert "describe_image" in result.content


def test_read_page_paginates_a_very_long_page_instead_of_silently_truncating():
    # Regression: silently cutting a page off at MAX_PAGE_CHARS lost whatever
    # came after with no way to get it back. offset/next_offset lets the
    # agent explicitly continue reading the same page instead.
    document = Document(
        file_name="long.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="x" * 20000, type="text", page_number=1)],
    )

    first = read_page([document], [{"file_name": "long.pdf", "page": 1}])

    assert f"call again with offset={MAX_PAGE_CHARS} for this page" in first.content
    assert first.sources[0].chunk.text == "x" * MAX_PAGE_CHARS

    second = read_page([document], [{"file_name": "long.pdf", "page": 1, "offset": MAX_PAGE_CHARS}])

    # Resumed from where the first call left off, not from the start again.
    assert f"call again with offset={MAX_PAGE_CHARS * 2} for this page" in second.content
    assert second.sources[0].chunk.text == "x" * MAX_PAGE_CHARS


def test_read_page_reports_an_offset_past_the_end_of_the_page():
    document = Document(
        file_name="short.pdf",
        file_type=".pdf",
        blocks=[Block(id="b1", text="hello", type="text", page_number=1)],
    )

    result = read_page([document], [{"file_name": "short.pdf", "page": 1, "offset": 100}])

    assert "offset 100 is beyond" in result.content
    assert result.sources == []


def test_read_page_batches_multiple_pages_across_documents():
    other_document = Document(
        file_name="other.pdf",
        file_type=".pdf",
        blocks=[Block(id="o1", text="Other document page one", type="text", page_number=1)],
    )

    result = read_page(
        [make_paged_document(), other_document],
        [
            {"file_name": "report.pdf", "page": 7},
            {"file_name": "other.pdf", "page": 1},
        ],
    )

    assert "=== report.pdf p.7 ===" in result.content
    assert "=== other.pdf p.1 ===" in result.content
    assert "Revenue table" in result.content
    assert "Other document page one" in result.content
    assert len(result.sources) == 2


def test_read_page_batch_isolates_a_bad_ref_from_the_good_ones():
    result = read_page(
        [make_paged_document()],
        [
            {"file_name": "report.pdf", "page": 7},
            {"file_name": "report.pdf", "page": 99},
        ],
    )

    assert "Revenue table" in result.content
    assert "no page 99" in result.content
    # Only the successful page becomes evidence.
    assert len(result.sources) == 1
    assert len(result.content) < 20000


class FakeVLMClient:
    def __init__(self, description="A bar chart showing quarterly revenue."):
        self.description = description
        self.calls = []

    def describe_image(self, image, prompt):
        self.calls.append((image, prompt))
        return self.description


def make_figure_document(file_name="deck.pdf", file_type="pdf", extra_blocks=()) -> Document:
    return Document(
        file_name=file_name,
        file_type=file_type,
        blocks=[
            Block(
                id="f1",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=3,
                bbox=(10.0, 10.0, 100.0, 100.0),
                metadata={"dataset_doc_id": f"q0001/{file_name}"},
            ),
            *extra_blocks,
        ],
    )


def test_limit_image_pixels_leaves_a_small_image_unchanged():
    image = Image.new("RGB", (100, 100))  # 10,000 px, well under the cap

    result = tools_module._limit_image_pixels(image, max_pixels=1_500_000)

    assert result is image


def test_limit_image_pixels_downscales_a_large_image():
    image = Image.new("RGB", (4000, 3000))  # 12,000,000 px

    result = tools_module._limit_image_pixels(image, max_pixels=1_500_000)

    # Independent width/height rounding can land a hair over the exact cap -
    # what matters is landing close to it, not exceeding the original by much.
    assert result.width * result.height <= 1_500_000 * 1.01
    assert result.width * result.height < image.width * image.height
    # Aspect ratio preserved (within integer rounding).
    assert abs(result.width / result.height - 4000 / 3000) < 0.01


def test_describe_image_returns_the_vlm_description(monkeypatch, tmp_path):
    source_path = tmp_path / "deck.pdf"
    source_path.write_bytes(b"fake pdf bytes")
    fake_image = Image.new("RGB", (10, 10))
    monkeypatch.setattr(tools_module, "extract_image_from_pdf", lambda path, page, bbox: fake_image)
    vlm = FakeVLMClient("A bar chart showing quarterly revenue.")

    result = describe_image(
        [make_figure_document()],
        {"deck.pdf": source_path},
        vlm,
        "deck.pdf",
        3,
        "What does the chart show?",
    )

    assert result.content == "A bar chart showing quarterly revenue."
    assert len(result.sources) == 1
    assert result.sources[0].chunk.metadata["dataset_doc_id"] == "q0001/deck.pdf"
    assert result.sources[0].chunk.metadata["page_number"] == 3
    # Small image, under MAX_VLM_PIXELS - passed through unchanged, same object.
    assert vlm.calls[0][0] is fake_image
    # The actual question must reach the VLM prompt, not a generic ask -
    # the whole point of this parameter (see DESCRIBE_IMAGE_PROMPT).
    assert "What does the chart show?" in vlm.calls[0][1]


def test_describe_image_reports_an_unknown_file():
    result = describe_image([], {}, FakeVLMClient(), "missing.pdf", 1, "What is this?")

    assert "no document named" in result.content


def test_describe_image_reports_no_images_on_a_text_only_page():
    result = describe_image(
        [make_paged_document()],
        {"report.pdf": Path("report.pdf")},
        FakeVLMClient(),
        "report.pdf",
        7,
        "What is this?",
    )

    assert "No images found" in result.content
    assert result.sources == []


def test_describe_image_reports_extraction_failure(monkeypatch, tmp_path):
    source_path = tmp_path / "deck.pdf"
    source_path.write_bytes(b"fake pdf bytes")
    monkeypatch.setattr(tools_module, "extract_image_from_pdf", lambda path, page, bbox: None)

    result = describe_image(
        [make_figure_document()],
        {"deck.pdf": source_path},
        FakeVLMClient(),
        "deck.pdf",
        3,
        "What is this?",
    )

    assert "could not extract" in result.content.lower()
    assert result.sources == []


def test_describe_image_labels_multiple_figures_on_the_same_page(monkeypatch, tmp_path):
    source_path = tmp_path / "deck.pdf"
    source_path.write_bytes(b"fake pdf bytes")
    second_figure = Block(
        id="f2",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        page_number=3,
        bbox=(20.0, 20.0, 200.0, 200.0),
        metadata={"dataset_doc_id": "q0001/deck.pdf"},
    )
    document = make_figure_document(extra_blocks=[second_figure])
    monkeypatch.setattr(
        tools_module,
        "extract_image_from_pdf",
        lambda path, page, bbox: Image.new("RGB", (10, 10)),
    )
    vlm = FakeVLMClient()

    result = describe_image(
        [document], {"deck.pdf": source_path}, vlm, "deck.pdf", 3, "What is this?"
    )

    assert "Image 1 on page 3:" in result.content
    assert "Image 2 on page 3:" in result.content
    assert len(result.sources) == 2


def test_describe_image_uses_pptx_shape_extraction(monkeypatch, tmp_path):
    source_path = tmp_path / "deck.pptx"
    source_path.write_bytes(b"fake pptx bytes")
    document = Document(
        file_name="deck.pptx",
        file_type="pptx",
        blocks=[
            Block(
                id="f1",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=2,
                metadata={"shape_index": 4, "dataset_doc_id": "q0001/deck.pptx"},
            ),
        ],
    )
    captured = {}

    def fake_extract_pptx(path, slide_number, shape_index):
        captured["args"] = (path, slide_number, shape_index)
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr(tools_module, "extract_image_from_pptx", fake_extract_pptx)
    vlm = FakeVLMClient("A photo of a server rack.")

    result = describe_image(
        [document], {"deck.pptx": source_path}, vlm, "deck.pptx", 2, "What is this?"
    )

    assert result.content == "A photo of a server rack."
    assert captured["args"] == (source_path, 2, 4)


def test_describe_image_opens_a_standalone_image_file_directly(monkeypatch, tmp_path):
    """A standalone image upload (ImageParser) is a one-block Document with
    no bbox/shape_index - describe_image must open the file itself rather
    than trying to render/crop it like a PDF or PPTX figure."""
    source_path = tmp_path / "photo.png"
    source_path.write_bytes(b"not a real png")
    document = Document(
        file_name="photo.png",
        file_type="png",
        blocks=[
            Block(
                id="block-1",
                text="[Image file: photo.png]",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                metadata={"block_type": "figure"},
            ),
        ],
    )
    opened_paths = []
    fake_image = Image.new("RGB", (10, 10))

    def fake_open(path):
        opened_paths.append(path)
        return fake_image

    monkeypatch.setattr(tools_module.Image, "open", fake_open)
    vlm = FakeVLMClient("A photo of a cat.")

    result = describe_image(
        [document], {"photo.png": source_path}, vlm, "photo.png", 1, "What is this?"
    )

    assert result.content == "A photo of a cat."
    assert opened_paths == [source_path]
    assert vlm.calls[0][0] is fake_image


def test_build_default_tools_always_includes_every_tool():
    tools = build_default_tools(FakeRetriever([]))

    # describe_image is conditional on a configured VLM backend (off by
    # default), so it's excluded here rather than covered by ALL_TOOL_NAMES
    # equality - see test_build_default_tools_includes_describe_image_when_vlm_is_configured.
    assert {tool.name for tool in tools} == set(ALL_TOOL_NAMES) - {"describe_image"}


def test_build_default_tools_includes_describe_image_when_vlm_is_configured(monkeypatch):
    monkeypatch.setattr(tools_module, "create_vlm_client", lambda: FakeVLMClient())

    tools = build_default_tools(FakeRetriever([]))

    assert "describe_image" in {tool.name for tool in tools}


def test_build_default_tools_run_python_description_carries_the_spreadsheet_schema():
    """Every observed run_python failure but two was a guessed sheet or column
    name, and the model writes code without calling list_documents first - so
    the layout has to be in the description it always sees."""
    xlsx_document = Document(
        file_name="inventory-data.xlsx",
        file_type=".xlsx",
        blocks=[
            Block(
                id="b1",
                text="id\tproduct\tstock\n1\tpen\t5",
                type="xlsx_sheet",
                metadata={"sheet_name": "Inventory", "max_row": 2},
            )
        ],
    )

    tools = build_default_tools(FakeRetriever([]), documents=[xlsx_document])
    description = next(tool for tool in tools if tool.name == "run_python").description

    assert "Inventory" in description
    assert "id, product, stock" in description


def test_build_default_tools_run_python_description_omits_schema_without_sheets():
    tools = build_default_tools(FakeRetriever([]), documents=[make_paged_document()])
    description = next(tool for tool in tools if tool.name == "run_python").description

    assert "Spreadsheet layout" not in description


def test_build_default_tools_run_python_description_lists_available_files():
    tools = build_default_tools(
        FakeRetriever([]),
        document_paths={"data.xlsx": Path("data.xlsx"), "notes.txt": Path("notes.txt")},
    )

    run_python_tool = next(tool for tool in tools if tool.name == "run_python")
    assert "data.xlsx" in run_python_tool.description
    assert "notes.txt" in run_python_tool.description


def test_build_default_tools_run_python_description_mentions_the_context_file():
    tools = build_default_tools(FakeRetriever([]))
    description = next(tool for tool in tools if tool.name == "run_python").description

    assert "_context.json" in description


def test_build_default_tools_run_python_description_covers_docx_and_pdf_text():
    # python-docx (added to the sandbox image) can read a DOCX table directly;
    # PDF has no structured reader there, but its extracted text is mounted
    # as a plain .txt file (see _write_document_text_files), so the
    # description must point at that instead of saying PDF can't be read.
    tools = build_default_tools(FakeRetriever([]))
    description = next(tool for tool in tools if tool.name == "run_python").description

    assert "python-docx" in description
    assert ".txt" in description
    assert "PDF" in description


def test_build_default_tools_run_python_description_covers_docx_paragraphs_too():
    # python-docx is allowed for both .tables and .paragraphs - a manual
    # section/keyword scan was briefly forbidden (it duplicated
    # search_documents and ballooned token usage) but the restriction was
    # lifted once run_python's evidence carried the real source document
    # (see _referenced_dataset_doc_id) instead of a generic "run_python"
    # sentinel eval metrics couldn't match against a question's doc_ids.
    tools = build_default_tools(FakeRetriever([]))
    description = next(tool for tool in tools if tool.name == "run_python").description

    assert ".paragraphs" in description
    assert ".tables" in description


def test_build_default_tools_feeds_search_results_into_run_python(monkeypatch, tmp_path):
    """The whole point of tracking accumulated_context: a search_documents
    call earlier in the same trajectory must be visible to a later
    run_python call, without any network path between the sandbox and the
    retriever."""
    results = [make_result("Found via search", 0.9)]
    tools = build_default_tools(FakeRetriever(results))

    search_tool = next(tool for tool in tools if tool.name == "search_documents")
    search_tool.handler(query="anything")

    captured = {}

    def fake_run_sandboxed_code(source_paths, **kwargs):
        captured["context"] = json.loads(source_paths["_context.json"].read_text(encoding="utf-8"))
        return SandboxResult(
            stdout="ok\n", stderr="", exit_code=0, timed_out=False, truncated=False
        )

    monkeypatch.setattr(tools_module, "run_sandboxed_code", fake_run_sandboxed_code)
    run_python_tool = next(tool for tool in tools if tool.name == "run_python")
    run_python_tool.handler(code="print('ok')")

    assert captured["context"]["previous_tool_results"] == [
        {"text": "Found via search", "metadata": {}}
    ]


def test_build_default_tools_search_schema_uses_the_given_default_top_k():
    tools = build_default_tools(FakeRetriever([]), default_top_k=3)

    search_tool = next(tool for tool in tools if tool.name == "search_documents")
    assert search_tool.parameters["properties"]["top_k"]["default"] == 3
