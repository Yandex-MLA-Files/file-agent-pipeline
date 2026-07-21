import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from file_agent.chunking import Chunk, chunk_document
from file_agent.document import Block, Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
from file_agent.rag import answer_indexed_documents, index_documents, load_documents
from file_agent.retrieval import SearchResult


class FakeEmbeddingModel:
    def encode(self, sentences):
        return [[float(len(sentence) % 7 + 1)] * 4 for sentence in sentences]


class FakeRetriever:
    def __init__(self):
        self.chunks = []

    def index(self, chunks):
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5):
        if not self.chunks:
            return []
        return [SearchResult(chunk=self.chunks[-1], score=1.0)]

    def clear(self):
        self.chunks = []


class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "Generated answer"


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


def test_retriever_index_and_search_create_spans(span_exporter):
    span_exporter.clear()
    retriever = LanceDBRetriever(
        embedding_model=FakeEmbeddingModel(),
        table_name="test_telemetry_chunks",
        semantic_min_score=-1.0,
    )
    chunks = [Chunk(id="c1", text="python markdown parser")]

    try:
        retriever.index(chunks)
        retriever.search("python", top_k=5)
    finally:
        retriever.clear()

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.retriever_index" in span_names
    assert "file_agent.retriever_search" in span_names


def test_load_documents_creates_span(tmp_path, span_exporter):
    span_exporter.clear()
    file_path = tmp_path / "note.md"
    file_path.write_text("# Title\n\nSome text")

    load_documents([file_path])

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.load_documents" in span_names


def test_index_documents_creates_span(span_exporter):
    span_exporter.clear()
    document = Document(
        file_name="note.md",
        file_type="md",
        blocks=[Block(id="b1", type="text", text="hello world")],
    )

    index_documents([document], FakeRetriever())

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.index_documents" in span_names


def test_answer_indexed_documents_creates_span(span_exporter):
    span_exporter.clear()
    retriever = FakeRetriever()
    retriever.index([Chunk(id="c1", text="hello world")])

    answer_indexed_documents(
        question="question?",
        llm_client=DummyLLM(),
        retriever=retriever,
        documents_count=1,
        chunks_count=1,
    )

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.answer_indexed_documents" in span_names


def test_parse_file_records_error_status_on_exception(tmp_path, span_exporter):
    span_exporter.clear()
    file_path = tmp_path / "note.xyz"
    file_path.write_text("unsupported")

    with pytest.raises(ValueError):
        parse_file(file_path)

    spans = span_exporter.get_finished_spans()
    parse_span = next(span for span in spans if span.name == "file_agent.parse_file")

    assert parse_span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in parse_span.events)
