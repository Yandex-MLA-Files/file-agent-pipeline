from file_agent.chunking import Chunk
from file_agent.rag import answer_indexed_documents_with_routing
from file_agent.retrieval import SearchResult
from file_agent.router import QueryType


class RoutedLLM:
    """Returns a router JSON response first, then a fixed answer for every later call.

    For a "complex" query_type this makes the planner call fail to parse JSON
    too (it also gets "Generated answer"), which is fine: plan_subqueries
    falls back to the original question as a single subquery.
    """

    def __init__(self, query_type: str):
        self.query_type = query_type
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            return f'{{"query_type": "{self.query_type}"}}'
        return "Generated answer"


class FakeRetriever:
    def __init__(self, chunk: Chunk | None):
        self.chunk = chunk
        self.search_calls = []

    def index(self, chunks):
        pass

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        if self.chunk is None:
            return []
        return [SearchResult(chunk=self.chunk, score=1.0)]

    def clear(self):
        pass


def _make_chunk() -> Chunk:
    return Chunk(id="chunk-1", text="Some indexed passage", metadata={"source_file": "notes.md"})


def test_answer_indexed_documents_with_routing_tags_response_with_query_type():
    llm_client = RoutedLLM(query_type="complex")
    retriever = FakeRetriever(chunk=_make_chunk())

    response = answer_indexed_documents_with_routing(
        question="Сравни данные из двух документов",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=2,
        chunks_count=5,
    )

    assert response.query_type == QueryType.COMPLEX
    assert response.answer == "Generated answer"
    assert llm_client.calls == 3  # classify + plan (fallback) + generate


def test_answer_indexed_documents_with_routing_still_runs_normal_rag_path():
    llm_client = RoutedLLM(query_type="simple")
    retriever = FakeRetriever(chunk=_make_chunk())

    response = answer_indexed_documents_with_routing(
        question="Что такое проект?",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=1,
        top_k=3,
    )

    assert response.query_type == QueryType.SIMPLE
    assert retriever.search_calls == [("Что такое проект?", 3)]
    assert response.documents_count == 1
    assert response.chunks_count == 1
