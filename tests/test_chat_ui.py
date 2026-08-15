from file_agent.chat_ui import (
    DEFAULT_CHAT_TITLE,
    chat_title_from_question,
    compact_sources,
    create_chat_session,
)
from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult


def make_result(
    chunk_id: str,
    text: str,
    metadata: dict,
    score: float = 0.8,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, text=text, metadata=metadata),
        score=score,
    )


def test_create_chat_session_uses_thread_id_and_has_no_messages():
    session = create_chat_session("thread-1")

    assert session == {
        "thread_id": "thread-1",
        "title": DEFAULT_CHAT_TITLE,
        "messages": [],
    }


def test_chat_title_uses_first_question_and_limits_its_length():
    assert chat_title_from_question("  Какие   условия расторжения?  ") == (
        "Какие условия расторжения?"
    )
    assert chat_title_from_question("Очень длинный вопрос о документе", max_length=16) == (
        "Очень длинный в…"
    )
    assert chat_title_from_question("   ") == DEFAULT_CHAT_TITLE


def test_compact_sources_keep_only_user_facing_location_and_short_excerpt():
    results = [
        make_result(
            "first",
            "Fallback chunk text",
            {
                "source_file": "report.pdf",
                "page_numbers": [3, 4, 5],
                "section": "Quarterly revenue",
                "_llm_context": "Revenue in the second quarter was 140 million rubles.",
                "internal_id": "must-not-leak",
            },
        ),
        make_result(
            "duplicate",
            "Duplicate from the same location",
            {
                "source_file": "report.pdf",
                "page_numbers": [3, 4, 5],
                "section": "Quarterly revenue",
            },
        ),
    ]

    assert compact_sources(results) == [
        {
            "file_name": "report.pdf",
            "location": "стр. 3–5 · раздел «Quarterly revenue»",
            "excerpt": "Revenue in the second quarter was 140 million rubles.",
        }
    ]


def test_compact_sources_support_slides_sheets_and_source_limit():
    results = [
        make_result(
            "slide",
            "Slide text",
            {"source_file": "deck.pptx", "slide_number": 7},
        ),
        make_result(
            "sheet",
            "Sheet text",
            {"source_file": "budget.xlsx", "sheet_name": "Summary"},
        ),
    ]

    assert compact_sources(results, max_sources=1) == [
        {
            "file_name": "deck.pptx",
            "location": "слайд 7",
            "excerpt": "Slide text",
        }
    ]
    assert compact_sources(results)[1]["location"] == "лист «Summary»"
