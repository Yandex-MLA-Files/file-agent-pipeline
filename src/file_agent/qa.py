import re

from file_agent.llm.base import LLMClient
from file_agent.retrieval import SearchResult

NO_CONTEXT_MESSAGE = "No relevant context was found in the document to answer the question."
CLARIFICATION_MESSAGE = "Please make the question more specific before searching the documents."
CLARIFICATION_MESSAGE_RU = "Пожалуйста, уточните вопрос перед поиском по документам."

QUERY_ROUTE_RETRIEVE = "retrieve"
QUERY_ROUTE_CLARIFY = "clarify"
CONTEXT_RELEVANT = "relevant"
CONTEXT_IRRELEVANT = "irrelevant"


def select_context_passages(
    results: list[SearchResult],
) -> list[tuple[SearchResult, str]]:
    """Return the unique passages that will actually be shown to the LLM.

    Retrieval operates on compact chunks, while split sections and tables may
    carry a larger parent passage in ``metadata["context"]``. Keeping this
    selection in one function prevents the QA prompt and exported HF contexts
    from drifting apart.
    """
    selected: list[tuple[SearchResult, str]] = []
    seen_passages: set[str] = set()

    for result in results:
        passage = result.chunk.metadata.get("context") or result.chunk.text
        if passage in seen_passages:
            continue
        seen_passages.add(passage)
        selected.append((result, passage))

    return selected


def build_context_from_results(results: list[SearchResult]) -> str:
    """Assemble the LLM context from search results (small-to-big retrieval).

    Chunks are sized for the embedding model, which makes them precise to
    retrieve but too short to answer from. When a chunk carries its parent
    passage in ``metadata["context"]`` the LLM reads that passage instead of
    the bare chunk; several chunks pointing at the same parent are collapsed
    so the prompt never repeats a passage.
    """
    context_parts: list[str] = []
    for index, (result, passage) in enumerate(select_context_passages(results), start=1):
        chunk = result.chunk
        metadata = ", ".join(
            f"{key}={value}" for key, value in chunk.metadata.items() if key != "context"
        )
        context_parts.append(
            f"[Chunk {index} | score={result.score:g} | metadata: {metadata}]\n{passage}"
        )

    return "\n\n".join(context_parts)


def build_qa_prompt(question: str, context: str) -> str:
    return (
        "Ответьте на вопрос, используя только приведённый ниже контекст.\n"
        "Напишите ответ только на том же языке, что и вопрос. Если вопрос "
        "задан на русском, отвечайте только на русском и не переходите на "
        "другие языки.\n"
        "Если контекста недостаточно, сообщите, что в документах недостаточно "
        "данных для ответа.\n"
        "При работе с таблицами и сравнениями сохраняйте принадлежность фактов "
        "к каждой сущности и не меняйте отношения местами.\n"
        "Дайте краткий ответ и укажите доступные метаданные источника.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос:\n{question}\n\n"
        "Ответ только на языке вопроса:"
    )


def build_query_analysis_prompt(question: str) -> str:
    return (
        "Classify whether the user's question is specific enough to search in uploaded "
        "documents.\n"
        "Return exactly one lowercase token and nothing else:\n"
        "- retrieve: the question contains enough meaning to attempt document retrieval;\n"
        "- clarify: the question is empty, only refers to missing prior context, or is too "
        "vague to form a useful search query.\n"
        "Prefer retrieve when uncertain. Do not answer the question.\n\n"
        f"Question:\n{question}\n\n"
        "Decision:"
    )


def parse_query_route(response: str) -> str:
    tokens = _decision_tokens(response)
    if QUERY_ROUTE_CLARIFY in tokens:
        return QUERY_ROUTE_CLARIFY
    return QUERY_ROUTE_RETRIEVE


def build_context_grading_prompt(question: str, context: str) -> str:
    return (
        "Decide whether the retrieved document context contains information that could "
        "help answer the question.\n"
        "Return exactly one lowercase token and nothing else:\n"
        "- relevant: at least one passage is meaningfully related to the question;\n"
        "- irrelevant: the passages are unrelated or provide no useful evidence.\n"
        "Do not answer the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Decision:"
    )


def parse_context_relevance(response: str) -> bool:
    tokens = _decision_tokens(response)
    # An unrecognized grader response falls back to the existing standard-RAG
    # behavior instead of discarding potentially useful retrieval results.
    explicitly_irrelevant = (
        CONTEXT_IRRELEVANT in tokens
        or "not_relevant" in tokens
        or {"not", CONTEXT_RELEVANT}.issubset(tokens)
    )
    return not explicitly_irrelevant


def build_query_rewrite_prompt(
    original_question: str,
    previous_query: str,
) -> str:
    return (
        "Rewrite the question as a concise semantic-search query for a document index.\n"
        "Preserve the original meaning, include important entities and constraints, and "
        "do not invent facts.\n"
        "Return only the rewritten query without a label, explanation, or quotation "
        "marks.\n\n"
        f"Original question:\n{original_question}\n\n"
        f"Previous search query:\n{previous_query}\n\n"
        "Rewritten query:"
    )


def normalize_rewritten_query(response: str, fallback: str) -> str:
    lines = [line.strip() for line in response.replace("```", "").splitlines() if line.strip()]
    if lines and lines[0].casefold() in {"text", "plaintext", "markdown"}:
        lines.pop(0)
    if not lines:
        return fallback

    query = re.sub(
        r"^(?:rewritten\s+query|search\s+query|query)\s*:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    ).strip(" \"'")
    return query or fallback


def build_clarification_message(question: str) -> str:
    if re.search(r"[А-Яа-яЁё]", question):
        return CLARIFICATION_MESSAGE_RU
    return CLARIFICATION_MESSAGE


def _decision_tokens(response: str) -> set[str]:
    return set(re.findall(r"[a-z_]+", response.casefold()))


def answer_question_with_context(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
) -> str:
    if not results:
        return NO_CONTEXT_MESSAGE

    context = build_context_from_results(results)
    prompt = build_qa_prompt(question, context)
    return llm_client.generate(prompt)
