from file_agent.chunking import Chunk
from file_agent.rag import (
    answer_documents,
    answer_files,
    answer_indexed_documents,
    answer_with_results,
)
from file_agent.retrieval import SearchResult


class DummyLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


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
