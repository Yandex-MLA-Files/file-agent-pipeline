from file_agent.chunking import Chunk
from file_agent.rag import (
    answer_indexed_documents_with_plan,
    answer_indexed_documents_with_routing,
    merge_search_results,
)
from file_agent.retrieval import SearchResult
from file_agent.router import QueryType


class ScriptedLLM:
    """Returns each response in order, one per generate() call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class SubqueryRetriever:
    """Returns different pre-set results per query, so merging is observable."""

    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self.results_by_query = results_by_query
        self.search_calls: list[tuple[str, int]] = []

    def index(self, chunks):
        pass

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return self.results_by_query.get(query, [])

    def clear(self):
        pass


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, text=f"text-{chunk_id}", metadata={"source_file": "notes.md"}),
        score=score,
    )


def test_answer_indexed_documents_with_plan_searches_each_subquery():
    llm_client = ScriptedLLM(
        [
            '{"subqueries": ["Доходы 2026", "Доходы 2025"]}',
            "Итоговый ответ",
        ]
    )
    retriever = SubqueryRetriever(
        {
            "Доходы 2026": [_result("a", 0.9)],
            "Доходы 2025": [_result("b", 0.8)],
        }
    )

    response = answer_indexed_documents_with_plan(
        question="Сравни доходы за 2026 и 2025 год",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=2,
    )

    assert response.answer == "Итоговый ответ"
    assert {result.chunk.id for result in response.sources} == {"a", "b"}
    assert retriever.search_calls == [("Доходы 2026", 5), ("Доходы 2025", 5)]


def test_answer_indexed_documents_with_plan_falls_back_to_single_query_on_bad_plan():
    llm_client = ScriptedLLM(["не json", "Ответ"])
    retriever = SubqueryRetriever({"Оригинальный вопрос": [_result("a", 1.0)]})

    response = answer_indexed_documents_with_plan(
        question="Оригинальный вопрос",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=1,
    )

    assert retriever.search_calls == [("Оригинальный вопрос", 5)]
    assert response.answer == "Ответ"


def test_answer_indexed_documents_with_routing_uses_plan_for_complex_query():
    llm_client = ScriptedLLM(
        [
            '{"query_type": "complex"}',
            '{"subqueries": ["A", "B"]}',
            "Финальный ответ",
        ]
    )
    retriever = SubqueryRetriever({"A": [_result("a", 0.5)], "B": [_result("b", 0.5)]})

    response = answer_indexed_documents_with_routing(
        question="A и B?",
        llm_client=llm_client,
        retriever=retriever,
        documents_count=1,
        chunks_count=2,
    )

    assert response.query_type == QueryType.COMPLEX
    assert retriever.search_calls == [("A", 5), ("B", 5)]
    assert response.answer == "Финальный ответ"


def test_merge_search_results_dedupes_by_chunk_id_keeping_best_score():
    results_by_subquery = [
        [_result("a", 0.4), _result("b", 0.9)],
        [_result("a", 0.7), _result("c", 0.6)],
    ]

    merged = merge_search_results(results_by_subquery, top_k=10)

    assert [result.chunk.id for result in merged] == ["b", "a", "c"]
    assert next(result.score for result in merged if result.chunk.id == "a") == 0.7


def test_merge_search_results_caps_at_top_k():
    results_by_subquery = [[_result(str(i), float(i)) for i in range(10)]]

    merged = merge_search_results(results_by_subquery, top_k=3)

    assert [result.chunk.id for result in merged] == ["9", "8", "7"]
