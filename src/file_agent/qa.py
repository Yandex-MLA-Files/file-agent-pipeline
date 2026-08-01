from file_agent.llm.base import LLMClient
from file_agent.retrieval import SearchResult

NO_CONTEXT_MESSAGE = "No relevant context was found in the document to answer the question."


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
