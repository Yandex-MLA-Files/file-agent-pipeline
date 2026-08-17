import logging
import os
from typing import Any

from file_agent.llm.base import LLMClient
from file_agent.retrieval import SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "No relevant context was found in the document to answer the question."

# ``QA_PROMPT`` selects the answering prompt: ``v2`` (default) asks for a
# complete, grounded answer with source citations; ``v1`` is the original short
# prompt, kept so earlier runs stay reproducible.
DEFAULT_QA_PROMPT_VERSION = "v2"

# Chunk metadata shown to the LLM as the passage header. Everything else
# (block ids, bounding boxes, retrieval internals) is noise for answering.
_CONTEXT_HEADER_KEYS = (
    "source_file",
    "doc_title",
    "page_number",
    "page_numbers",
    "slide_number",
    "sheet_name",
    "heading_path",
    "section",
    "block_type",
    "time_start",
)


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


def qa_prompt_version() -> str:
    version = (os.getenv("QA_PROMPT") or DEFAULT_QA_PROMPT_VERSION).strip().lower()
    if version not in ("v1", "v2"):
        raise ValueError(f"QA_PROMPT must be 'v1' or 'v2', got {version!r}")
    return version


def build_context_from_results(results: list[SearchResult]) -> str:
    """Assemble the LLM context from search results (small-to-big retrieval).

    Chunks are sized for the embedding model, which makes them precise to
    retrieve but too short to answer from. When a chunk carries its parent
    passage in ``metadata["context"]`` the LLM reads that passage instead of
    the bare chunk; several chunks pointing at the same parent are collapsed
    so the prompt never repeats a passage.
    """
    version = qa_prompt_version()
    context_parts: list[str] = []
    for index, (result, passage) in enumerate(select_context_passages(results), start=1):
        chunk = result.chunk
        if version == "v1":
            metadata = ", ".join(
                f"{key}={value}" for key, value in chunk.metadata.items() if key != "context"
            )
            context_parts.append(
                f"[Chunk {index} | score={result.score:g} | metadata: {metadata}]\n{passage}"
            )
        else:
            context_parts.append(
                f"[Источник {index} | {_context_header(chunk.metadata)}]\n{passage}"
            )

    return "\n\n".join(context_parts)


def _context_header(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _CONTEXT_HEADER_KEYS:
        value = metadata.get(key)
        if value in (None, "", [], ()):
            continue
        if key == "page_numbers" and metadata.get("page_number") is not None and len(value) == 1:
            continue
        if key == "page_number" and metadata.get("page_numbers") not in (None, [], ()):
            if len(metadata["page_numbers"]) > 1:
                continue
        if key == "section" and metadata.get("heading_path"):
            continue
        if isinstance(value, (list, tuple)):
            value = (
                " > ".join(str(item) for item in value)
                if key == "heading_path"
                else ", ".join(str(item) for item in value)
            )
        parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else "metadata: none"


def build_qa_prompt(question: str, context: str) -> str:
    if qa_prompt_version() == "v1":
        return _build_qa_prompt_v1(question, context)
    return _build_qa_prompt_v2(question, context)


def _build_qa_prompt_v1(question: str, context: str) -> str:
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


def _build_qa_prompt_v2(question: str, context: str) -> str:
    return (
        "Вы отвечаете на вопросы строго по фрагментам документов, приведённым ниже.\n"
        "Правила:\n"
        "1. Используйте только сведения из контекста; ничего не добавляйте от себя. "
        "Если контекст не содержит ответа, прямо напишите, что в документах нет "
        "информации для ответа (и коротко укажите, что в них есть по теме).\n"
        "2. Ответ должен быть полным: перечислите все относящиеся к вопросу факты, "
        "числа, названия, условия и определения из контекста, ничего не пропуская. "
        "Числа, даты и единицы измерения приводите точно как в источнике.\n"
        "3. При сравнениях и таблицах сохраняйте принадлежность фактов к каждой "
        "сущности и не меняйте отношения местами. Если в контексте есть таблица, "
        "берите значения из нужной строки и столбца.\n"
        "4. Отвечайте на языке вопроса (на русский вопрос — только по-русски), "
        "связным текстом без вводных фраз о контексте.\n"
        "5. В конце укажите источники: файл и страницу/раздел из заголовков фрагментов.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос:\n{question}\n\n"
        "Ответ:"
    )


def answer_question_with_context(
    question: str,
    results: list[SearchResult],
    llm_client: LLMClient,
) -> str:
    with tracer.start_as_current_span("file_agent.answer_question_with_context") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.result_count", len(results))

        if not results:
            logger.info("No search results for question %r, skipping LLM call", question)
            return NO_CONTEXT_MESSAGE

        context = build_context_from_results(results)
        prompt = build_qa_prompt(question, context)
        span.set_attribute("file_agent.prompt_length", len(prompt))

        logger.info("Answering question %r with %d context result(s)", question, len(results))
        return llm_client.generate(prompt)
