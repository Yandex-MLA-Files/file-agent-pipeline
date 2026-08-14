import fitz
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.rag import (
    answer_documents,
    answer_files,
    answer_indexed_documents,
    answer_with_results,
    resolve_history_turns,
    resolve_max_tool_rounds,
    resolve_rag_mode,
)
from file_agent.retrieval import SearchResult
from file_agent.vlm.base import VLMClient


class DummyLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


class ToolSequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt: str) -> str:
        raise AssertionError("generate must not be used in tool_agent mode")

    def chat_with_tools(self, messages, tools, tool_choice="auto"):
        if not self.responses:
            raise AssertionError("Unexpected tool-calling LLM invocation")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self):
        self.chunks = []
        self.index_calls = 0
        self.search_calls = []

    def index(self, chunks):
        self.index_calls += 1
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        if not self.chunks:
            return []
        return [SearchResult(chunk=self.chunks[-1], score=1.0)]

    def clear(self):
        self.chunks = []


class StubVLM(VLMClient):
    def __init__(self):
        self.calls = 0

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        return "The visual shows an increase."


def test_answer_files_runs_full_rag_pipeline(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Project\n\nThe project parses markdown files.",
        encoding="utf-8",
    )
    llm_client = DummyLLM()
    retriever = FakeRetriever()

    response = answer_files(
        file_paths=[file_path],
        question="What parses markdown?",
        llm_client=llm_client,
        retriever=retriever,
    )

    assert response.answer == "Generated answer"
    assert response.documents_count == 1
    assert response.chunks_count == 1
    assert [source.chunk.id for source in response.sources] == ["block-1-chunk-1"]
    assert retriever.index_calls == 1
    assert retriever.search_calls == [("What parses markdown?", 5)]
    assert len(llm_client.prompts) == 1
    assert "The project parses markdown files." in llm_client.prompts[0]


def test_answer_documents_uses_no_context_message_without_matches():
    llm_client = DummyLLM()
    retriever = FakeRetriever()

    response = answer_documents(
        documents=[],
        question="What is the answer?",
        llm_client=llm_client,
        retriever=retriever,
    )

    assert "No relevant context" in response.answer
    assert response.sources == []
    assert response.documents_count == 0
    assert response.chunks_count == 0
    assert llm_client.prompts == []


def test_chunk_documents_combines_multiple_documents(tmp_path):
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("First document text", encoding="utf-8")
    second_path.write_text("Second document text", encoding="utf-8")

    response = answer_files(
        file_paths=[first_path, second_path],
        question="Second document",
        llm_client=DummyLLM(),
        retriever=FakeRetriever(),
    )

    assert response.documents_count == 2
    assert response.chunks_count == 2
    assert response.sources[0].chunk.metadata["source_file"] == "second.md"


def test_answer_indexed_documents_does_not_reindex_chunks():
    retriever = FakeRetriever()
    retriever.index([])

    response = answer_indexed_documents(
        question="Question",
        llm_client=DummyLLM(),
        retriever=retriever,
        documents_count=2,
        chunks_count=10,
    )

    assert retriever.index_calls == 1
    assert retriever.search_calls == [("Question", 5)]
    assert response.documents_count == 2
    assert response.chunks_count == 10


def test_answer_with_results_uses_precomputed_search_results():
    llm_client = DummyLLM()
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="Precomputed context",
                metadata={"source_file": "notes.md"},
            ),
            score=0.75,
        )
    ]

    response = answer_with_results(
        question="What is the context?",
        results=results,
        llm_client=llm_client,
        documents_count=1,
        chunks_count=1,
    )

    assert response.answer == "Generated answer"
    assert response.sources is results
    assert response.documents_count == 1
    assert response.chunks_count == 1
    assert len(llm_client.prompts) == 1
    assert "Precomputed context" in llm_client.prompts[0]


def test_answer_indexed_documents_can_use_tool_agent_mode():
    retriever = FakeRetriever()
    retriever.index(
        [
            Chunk(
                id="chunk-1",
                text="Tool context",
                metadata={"source_file": "notes.md"},
            )
        ]
    )
    llm_client = ToolSequenceLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "tool context"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Tool agent answer"),
        ]
    )

    response = answer_indexed_documents(
        question="What is the context?",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=1,
        top_k=3,
        mode="tool_agent",
    )

    assert response.answer == "Tool agent answer"
    assert response.search_queries == ["tool context"]
    assert retriever.search_calls == [("tool context", 3)]
    assert llm_client.responses == []


def test_answer_indexed_documents_passes_visual_runtime_dependencies():
    pdf = fitz.open()
    page = pdf.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(20, 20, 180, 180), fill=(0, 0, 1))
    pdf_bytes = pdf.tobytes()
    pdf.close()

    document = Document(
        file_name="chart.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="chart-1",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                bbox=(20, 20, 180, 180),
            )
        ],
    )
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put("chart.pdf", pdf_bytes)
    vlm_client = StubVLM()
    llm_client = ToolSequenceLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "chart.pdf",
                            "visual_id": "chart-1",
                            "question": "What trend is shown?",
                        },
                        "id": "visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The chart shows an increase [chart.pdf, page 1]."),
        ]
    )

    response = answer_indexed_documents(
        question="What trend is shown?",
        llm_client=llm_client,
        retriever=FakeRetriever(),
        documents_count=1,
        chunks_count=1,
        documents=[document],
        mode="tool_agent",
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    assert response.sources[0].chunk.metadata["visual_id"] == "chart-1"
    assert response.tool_calls[0]["name"] == "analyze_document_visual"
    assert vlm_client.calls == 1


def test_rag_mode_and_tool_round_limit_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "tool_agent")
    monkeypatch.setenv("RAG_MAX_TOOL_ROUNDS", "4")

    assert resolve_rag_mode() == "tool_agent"
    assert resolve_max_tool_rounds() == 4


def test_history_turn_limit_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_HISTORY_TURNS", "5")

    assert resolve_history_turns() == 5


@pytest.mark.parametrize("mode", ["unknown", "agent"])
def test_resolve_rag_mode_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="Unsupported RAG_MODE"):
        resolve_rag_mode(mode)


@pytest.mark.parametrize("value", [0, -1])
def test_resolve_max_tool_rounds_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_max_tool_rounds(value)


def test_resolve_max_tool_rounds_rejects_non_integer_environment_value(monkeypatch):
    monkeypatch.setenv("RAG_MAX_TOOL_ROUNDS", "many")

    with pytest.raises(ValueError, match="must be an integer"):
        resolve_max_tool_rounds()


@pytest.mark.parametrize("value", [0, -1])
def test_resolve_history_turns_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_history_turns(value)


def test_resolve_history_turns_rejects_non_integer_environment_value(monkeypatch):
    monkeypatch.setenv("RAG_HISTORY_TURNS", "many")

    with pytest.raises(ValueError, match="must be an integer"):
        resolve_history_turns()


def test_standard_mode_does_not_read_tool_round_setting(monkeypatch):
    monkeypatch.setenv("RAG_MAX_TOOL_ROUNDS", "many")
    llm_client = DummyLLM()
    results = [
        SearchResult(
            chunk=Chunk(id="chunk-1", text="Context", metadata={}),
            score=1.0,
        )
    ]

    response = answer_with_results(
        question="Question",
        results=results,
        llm_client=llm_client,
        documents_count=1,
        chunks_count=1,
        mode="standard",
    )

    assert response.answer == "Generated answer"


def test_tool_agent_mode_requires_tool_calling_llm():
    with pytest.raises(TypeError, match="does not support native tool calling"):
        answer_indexed_documents(
            question="Question",
            llm_client=DummyLLM(),
            retriever=FakeRetriever(),
            documents_count=0,
            chunks_count=0,
            mode="tool_agent",
        )
