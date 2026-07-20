from file_agent.llm.base import LLMClient
from file_agent.retrieval import SearchResult

NO_CONTEXT_MESSAGE = "No relevant context was found in the document to answer the question."


def build_context_from_results(results: list[SearchResult]) -> str:
    context_parts: list[str] = []

    for index, result in enumerate(results, start=1):
        metadata = ", ".join(f"{key}={value}" for key, value in result.chunk.metadata.items())
        context_parts.append(
            f"[Chunk {index} | score={result.score:g} | metadata: {metadata}]\n{result.chunk.text}"
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
