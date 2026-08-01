from file_agent.chunking import Chunk
from file_agent.document import Block, Document
from file_agent.rag_graph import (
    IngestionContext,
    QAContext,
    build_agentic_qa_graph,
    build_ingestion_graph,
    build_qa_graph,
)
from file_agent.retrieval import SearchResult


class DummyLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
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


def test_ingestion_graph_loads_chunks_and_indexes_files(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Project\n\nLangGraph runs the workflow.", encoding="utf-8")
    retriever = FakeRetriever()

    state = build_ingestion_graph().invoke(
        {"file_paths": [file_path]},
        context=IngestionContext(retriever=retriever),
    )

    assert state["documents_count"] == 1
    assert state["chunks_count"] == 1
    assert state["documents"][0].file_name == "notes.md"
    assert retriever.index_calls == 1
    assert retriever.chunks == state["chunks"]


def test_ingestion_graph_accepts_preparsed_documents():
    document = Document(
        file_name="notes.md",
        file_type=".md",
        blocks=[
            Block(
                id="block-1",
                text="Already parsed text",
                type="text",
                metadata={"source_file": "notes.md"},
            )
        ],
    )
    retriever = FakeRetriever()

    state = build_ingestion_graph().invoke(
        {"documents": [document]},
        context=IngestionContext(retriever=retriever),
    )

    assert state["documents"] == [document]
    assert state["documents_count"] == 1
    assert state["chunks_count"] == 1
    assert retriever.index_calls == 1


def test_qa_graph_retrieves_and_generates_answer():
    retriever = FakeRetriever()
    retriever.index(
        [
            Chunk(
                id="chunk-1",
                text="LangGraph runs the workflow.",
                metadata={"source_file": "notes.md"},
            )
        ]
    )
    llm_client = DummyLLM()

    state = build_qa_graph().invoke(
        {
            "question": "What runs the workflow?",
            "top_k": 3,
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
        ),
    )

    assert retriever.search_calls == [("What runs the workflow?", 3)]
    assert state["response"].answer == "Generated answer"
    assert state["response"].sources == state["results"]
    assert "LangGraph runs the workflow." in llm_client.prompts[0]


def test_qa_graph_uses_precomputed_results_without_retrieval():
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
    llm_client = DummyLLM()

    state = build_qa_graph().invoke(
        {
            "question": "What is the context?",
            "results": results,
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=QAContext(llm_client=llm_client),
    )

    assert state["response"].sources is results
    assert state["response"].answer == "Generated answer"
    assert "Precomputed context" in llm_client.prompts[0]


def test_agentic_qa_graph_generates_from_relevant_context():
    retriever = FakeRetriever()
    retriever.index(
        [
            Chunk(
                id="chunk-1",
                text="The project deadline is Friday.",
                metadata={"source_file": "notes.md"},
            )
        ]
    )
    llm_client = SequenceLLM(
        [
            "retrieve",
            "relevant",
            "The deadline is Friday.",
        ]
    )

    state = build_agentic_qa_graph().invoke(
        {
            "question": "What is the project deadline?",
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
        ),
    )

    response = state["response"]
    assert response.answer == "The deadline is Friday."
    assert response.retry_count == 0
    assert response.search_queries == ["What is the project deadline?"]
    assert response.stop_reason == "answer_generated"
    assert retriever.search_calls == [("What is the project deadline?", 5)]
    assert llm_client.responses == []


def test_agentic_qa_graph_rewrites_irrelevant_query_and_retries():
    retriever = FakeRetriever()
    retriever.index(
        [
            Chunk(
                id="chunk-1",
                text="The implementation finishes on Friday.",
                metadata={"source_file": "plan.md"},
            )
        ]
    )
    llm_client = SequenceLLM(
        [
            "retrieve",
            "irrelevant",
            "implementation completion deadline",
            "relevant",
            "The implementation finishes on Friday.",
        ]
    )

    state = build_agentic_qa_graph().invoke(
        {
            "question": "When is it done?",
            "top_k": 3,
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
            max_retries=2,
        ),
    )

    response = state["response"]
    assert retriever.search_calls == [
        ("When is it done?", 3),
        ("implementation completion deadline", 3),
    ]
    assert response.answer == "The implementation finishes on Friday."
    assert response.retry_count == 1
    assert response.search_queries == [
        "When is it done?",
        "implementation completion deadline",
    ]
    assert llm_client.responses == []


def test_agentic_qa_graph_stops_after_bounded_retries_without_context():
    retriever = FakeRetriever()
    llm_client = SequenceLLM(
        [
            "retrieve",
            "first rewritten query",
            "second rewritten query",
        ]
    )

    state = build_agentic_qa_graph().invoke(
        {
            "question": "What is the answer?",
            "documents_count": 0,
            "chunks_count": 0,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
            max_retries=2,
        ),
    )

    response = state["response"]
    assert retriever.search_calls == [
        ("What is the answer?", 5),
        ("first rewritten query", 5),
        ("second rewritten query", 5),
    ]
    assert response.answer.startswith("No relevant context")
    assert response.sources == []
    assert response.retry_count == 2
    assert response.stop_reason == "insufficient_context"
    assert llm_client.responses == []


def test_agentic_qa_graph_asks_to_clarify_without_retrieval():
    retriever = FakeRetriever()
    llm_client = SequenceLLM(["clarify"])

    state = build_agentic_qa_graph().invoke(
        {
            "question": "What about it?",
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=QAContext(
            llm_client=llm_client,
            retriever=retriever,
        ),
    )

    response = state["response"]
    assert "more specific" in response.answer
    assert response.sources == []
    assert response.stop_reason == "clarification_needed"
    assert retriever.search_calls == []
    assert llm_client.responses == []
