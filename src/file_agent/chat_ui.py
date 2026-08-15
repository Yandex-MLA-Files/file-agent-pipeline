"""Small, serializable view models for the Streamlit chat UI."""

from typing import Any, TypedDict

from file_agent.retrieval import SearchResult

DEFAULT_CHAT_TITLE = "Новый чат"


class CompactSource(TypedDict):
    file_name: str
    location: str
    excerpt: str


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    sources: list[CompactSource]


class ChatSession(TypedDict):
    thread_id: str
    title: str
    messages: list[ChatMessage]


def create_chat_session(thread_id: str) -> ChatSession:
    """Create a chat record whose id can be passed directly to LangGraph."""
    return {
        "thread_id": thread_id,
        "title": DEFAULT_CHAT_TITLE,
        "messages": [],
    }


def chat_title_from_question(question: str, max_length: int = 44) -> str:
    """Build a compact sidebar title from the first user question."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    title = " ".join(question.split())
    if not title:
        return DEFAULT_CHAT_TITLE
    if len(title) <= max_length:
        return title
    return title[: max_length - 1].rstrip(" .,;:!?-") + "…"


def compact_sources(
    results: list[SearchResult],
    *,
    max_sources: int = 5,
    excerpt_length: int = 220,
) -> list[CompactSource]:
    """Convert retrieval results into concise, safe-to-persist UI citations."""
    if max_sources < 1:
        return []

    sources: list[CompactSource] = []
    seen: set[tuple[str, str]] = set()

    for result in results:
        metadata = result.chunk.metadata
        file_name = str(metadata.get("source_file") or "Документ")
        location = _source_location(metadata)
        key = (file_name.casefold(), location.casefold())
        if key in seen:
            continue

        context = metadata.get("_llm_context") or metadata.get("context")
        excerpt = _shorten_text(
            context if isinstance(context, str) else result.chunk.text,
            excerpt_length,
        )
        sources.append(
            {
                "file_name": file_name,
                "location": location,
                "excerpt": excerpt,
            }
        )
        seen.add(key)
        if len(sources) >= max_sources:
            break

    return sources


def _source_location(metadata: dict[str, Any]) -> str:
    parts: list[str] = []

    pages = _integer_coordinates(metadata, "page_number", "page_numbers")
    slides = _integer_coordinates(metadata, "slide_number", "slide_numbers")
    sheets = _text_coordinates(metadata, "sheet_name", "sheet_names")

    if pages:
        parts.append(_format_number_coordinates(pages, "стр."))
    elif slides:
        parts.append(_format_number_coordinates(slides, "слайд"))
    elif sheets:
        quoted_sheets = ", ".join(f"«{sheet}»" for sheet in sheets)
        parts.append(f"лист {quoted_sheets}")

    section = metadata.get("section")
    if isinstance(section, str) and section.strip():
        parts.append(f"раздел «{_shorten_text(section, 60)}»")

    return " · ".join(parts)


def _integer_coordinates(
    metadata: dict[str, Any],
    singular_key: str,
    plural_key: str,
) -> list[int]:
    raw_values = metadata.get(plural_key)
    if not isinstance(raw_values, (list, tuple, set)):
        raw_values = [metadata.get(singular_key)]

    values = {
        value
        for value in raw_values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    return sorted(values)


def _text_coordinates(
    metadata: dict[str, Any],
    singular_key: str,
    plural_key: str,
) -> list[str]:
    raw_values = metadata.get(plural_key)
    if not isinstance(raw_values, (list, tuple, set)):
        raw_values = [metadata.get(singular_key)]

    values = {value.strip() for value in raw_values if isinstance(value, str) and value.strip()}
    return sorted(values, key=str.casefold)


def _format_number_coordinates(values: list[int], label: str) -> str:
    if len(values) == 1:
        return f"{label} {values[0]}"
    if values == list(range(values[0], values[-1] + 1)):
        return f"{label} {values[0]}–{values[-1]}"
    return f"{label} {', '.join(str(value) for value in values)}"


def _shorten_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if max_length < 1 or len(normalized) <= max_length:
        return normalized
    if max_length == 1:
        return "…"
    return normalized[: max_length - 1].rstrip() + "…"
