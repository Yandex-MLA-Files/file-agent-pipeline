import base64
import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from file_agent.chunking import Chunk, chunk_document
from file_agent.document import Block, Document
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.openai_client import OpenAILLMClient
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
from file_agent.rag import answer_indexed_documents, index_documents, ingest_files, load_documents
from file_agent.retrieval import SearchResult
from file_agent.telemetry import _create_langfuse_exporter


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
    search_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "file_agent.retriever_search"
    )
    assert search_span.attributes["langfuse.observation.type"] == "retriever"
    assert json.loads(search_span.attributes["langfuse.observation.input"])["query"] == "python"


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


def test_langgraph_ingestion_keeps_index_span(tmp_path, span_exporter):
    span_exporter.clear()
    file_path = tmp_path / "note.md"
    file_path.write_text("# Title\n\nSome text")

    ingest_files([file_path], FakeRetriever())

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.load_documents" in span_names
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
    answer_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "file_agent.answer_indexed_documents"
    )
    assert answer_span.attributes["langfuse.observation.type"] == "agent"
    assert (
        json.loads(answer_span.attributes["langfuse.observation.input"])["question"] == "question?"
    )
    assert json.loads(answer_span.attributes["langfuse.observation.output"])["answer"] == (
        "Generated answer"
    )


def test_llm_generation_has_langfuse_attributes(span_exporter):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Generated answer"))],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=3,
                    total_tokens=13,
                ),
            )

    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    span_exporter.clear()

    OpenAILLMClient(client=fake_openai, model="test-model").generate("Question")

    generation = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "file_agent.llm_generate"
    )
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert generation.attributes["langfuse.observation.model.name"] == "test-model"
    assert json.loads(generation.attributes["langfuse.observation.usage_details"]) == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


def test_langfuse_exporter_is_disabled_without_configuration(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    assert _create_langfuse_exporter() is None


def test_langfuse_exporter_uses_http_endpoint_and_basic_auth(monkeypatch):
    calls = []

    def fake_exporter(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000/")
    monkeypatch.setattr("file_agent.telemetry.OTLPHTTPSpanExporter", fake_exporter)

    exporter = _create_langfuse_exporter()

    assert exporter is not None
    assert calls[0]["endpoint"] == "http://localhost:3000/api/public/otel/v1/traces"
    credentials = base64.b64encode(b"pk-test:sk-test").decode("ascii")
    assert calls[0]["headers"] == {
        "Authorization": f"Basic {credentials}",
        "x-langfuse-ingestion-version": "4",
    }


def test_langfuse_exporter_rejects_partial_configuration(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    with pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"):
        _create_langfuse_exporter()


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


def _make_text_pdf(path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from a real PDF page with a text layer.")
    doc.save(path)
    doc.close()


def test_analyze_pdf_creates_span(tmp_path, span_exporter):
    from file_agent.parsers.routing import analyze_pdf

    span_exporter.clear()
    pdf_path = tmp_path / "routing.pdf"
    _make_text_pdf(pdf_path)

    analyze_pdf(pdf_path)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.analyze_pdf" in span_names


def test_docling_parse_creates_span(tmp_path, span_exporter):
    from file_agent.parsers.docling_parser import DoclingParser

    span_exporter.clear()
    pdf_path = tmp_path / "docling.pdf"
    _make_text_pdf(pdf_path)

    DoclingParser().parse(pdf_path)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.docling_parse" in span_names


def test_enhance_document_creates_span(tmp_path, span_exporter):
    from file_agent.document import Block, BlockType
    from file_agent.document import Document as FADocument
    from file_agent.parsers.enhancer import DocumentEnhancer
    from file_agent.vlm.base import VLMClient

    class StubVLMClient(VLMClient):
        def describe_image(self, image, prompt: str) -> str:
            return "stub description"

    span_exporter.clear()
    pdf_path = tmp_path / "enhance.pdf"
    _make_text_pdf(pdf_path)

    figure = Block(
        id="f1",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        page_number=1,
        bbox=(50.0, 50.0, 200.0, 200.0),
    )
    document = FADocument(file_name="enhance.pdf", file_type="pdf", blocks=[figure])

    DocumentEnhancer(vlm_client=StubVLMClient()).enhance(document, pdf_path)

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.enhance_document" in span_names


def test_vlm_describe_image_creates_span(span_exporter):
    from types import SimpleNamespace

    from PIL import Image

    from file_agent.vlm.openai_compatible import OpenAICompatibleVLMClient

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="fake description"))]
            )

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    span_exporter.clear()
    client = OpenAICompatibleVLMClient(base_url="http://unused", model="test-vlm")
    client.client = FakeOpenAI()

    image = Image.new("RGB", (10, 10))
    description = client.describe_image(image, "describe this")

    assert description == "fake description"
    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert "file_agent.vlm_describe_image" in span_names
