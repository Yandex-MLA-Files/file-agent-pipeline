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
        "Answer the question using only the context below.\n"
        "If the context does not contain enough information, say that the "
        "documents do not contain enough data to answer.\n"
        "Keep the answer concise and mention relevant source metadata when "
        "it is present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
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
