from file_agent.llm.base import LLMClient
from file_agent.retrieval import SearchResult

NO_CONTEXT_MESSAGE = "No relevant context was found in the document to answer the question."


def build_context_from_results(results: list[SearchResult]) -> str:
    """Assemble the LLM context from search results (small-to-big retrieval).

    Chunks are sized for the embedding model, which makes them precise to
    retrieve but too short to answer from. When a chunk carries its parent
    passage in ``metadata["context"]`` the LLM reads that passage instead of
    the bare chunk; several chunks pointing at the same parent are collapsed
    so the prompt never repeats a passage.
    """
    context_parts: list[str] = []
    seen_passages: set[str] = set()
    index = 0

    for result in results:
        chunk = result.chunk
        passage = chunk.metadata.get("context") or chunk.text
        if passage in seen_passages:
            continue
        seen_passages.add(passage)

        index += 1
        metadata = ", ".join(
            f"{key}={value}" for key, value in chunk.metadata.items() if key != "context"
        )
        context_parts.append(
            f"[Chunk {index} | score={result.score:g} | metadata: {metadata}]\n{passage}"
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
