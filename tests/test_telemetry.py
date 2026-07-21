import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from file_agent.chunking import chunk_document
from file_agent.document import Block, Document
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context


@pytest.fixture(scope="module")
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_parse_file_creates_span(tmp_path, span_exporter):
    span_exporter.clear()
    file_path = tmp_path / "note.md"
    file_path.write_text("# Title\n\nSome text")

    parse_file(file_path)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.parse_file" in span_names


def test_chunk_document_creates_span(span_exporter):
    span_exporter.clear()
    document = Document(
        file_name="note.md",
        file_type="md",
        blocks=[Block(id="b1", type="text", text="hello world")],
    )

    chunk_document(document)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.chunk_document" in span_names


def test_answer_question_with_context_creates_span_without_results(span_exporter):
    span_exporter.clear()

    answer_question_with_context("question?", [], llm_client=None)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.answer_question_with_context" in span_names
