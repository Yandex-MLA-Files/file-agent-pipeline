from file_agent.chunking import Chunk
from file_agent.qa import (
    NO_CONTEXT_MESSAGE,
    answer_question_with_context,
    build_context_from_results,
    build_qa_prompt,
)
from file_agent.retrieval import SearchResult


class DummyLLMClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


def test_build_context_from_results_preserves_chunk_texts():
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="First chunk text",
                metadata={"source_file": "notes.md", "block_id": "block-1"},
            ),
            score=2.0,
        ),
        SearchResult(
            chunk=Chunk(
                id="chunk-2",
                text="Second chunk text",
                metadata={"source_file": "notes.md", "block_id": "block-2"},
            ),
            score=1.0,
        ),
    ]

    context = build_context_from_results(results)

    assert "First chunk text" in context
    assert "Second chunk text" in context
    assert "source_file=notes.md" in context
    # Retrieval internals are not shown to the LLM in the default (v2) format.
    assert "block_id=block-1" not in context
    assert context.startswith("[Источник 1 | source_file=notes.md]")


def test_build_context_v1_keeps_all_metadata(monkeypatch):
    monkeypatch.setenv("QA_PROMPT", "v1")
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="First chunk text",
                metadata={"source_file": "notes.md", "block_id": "block-1"},
            ),
            score=2.0,
        )
    ]

    context = build_context_from_results(results)

    assert "block_id=block-1" in context
    assert context.startswith("[Chunk 1 | score=2 | metadata:")


def test_build_context_header_shows_heading_path_and_pages():
    results = [
        SearchResult(
            chunk=Chunk(
                id="c1",
                text="body",
                metadata={
                    "source_file": "doc.pdf",
                    "page_numbers": [3, 4],
                    "page_number": 3,
                    "heading_path": ["Chapter 1", "Section 2"],
                    "section": "Section 2",
                    "block_ids": ["b1"],
                    "bbox": (0, 0, 1, 1),
                },
            ),
            score=1.0,
        )
    ]

    context = build_context_from_results(results)

    header = context.split("\n", 1)[0]
    assert "page_numbers=3, 4" in header
    assert "heading_path=Chapter 1 > Section 2" in header
    assert "section=" not in header
    assert "bbox" not in header and "block_ids" not in header


def test_build_context_from_results_deduplicates_parent_passages():
    parent = "Full section used to answer the question"
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="First fragment",
                metadata={"context": parent, "page_number": 1},
            ),
            score=1.0,
        ),
        SearchResult(
            chunk=Chunk(
                id="chunk-2",
                text="Second fragment",
                metadata={"context": parent, "page_number": 1},
            ),
            score=0.5,
        ),
    ]

    context = build_context_from_results(results)

    assert context.count(parent) == 1
    assert "First fragment" not in context
    assert "Second fragment" not in context


def test_build_qa_prompt_contains_question_and_context():
    prompt = build_qa_prompt(
        question="What is the document about?",
        context="Document context",
    )

    assert "What is the document about?" in prompt
    assert "Document context" in prompt
    assert "только сведения из контекста" in prompt
    assert "на языке вопроса" in prompt
    assert "не меняйте отношения местами" in prompt
    assert prompt.endswith("Ответ:")


def test_build_qa_prompt_v1_is_still_available(monkeypatch):
    monkeypatch.setenv("QA_PROMPT", "v1")

    prompt = build_qa_prompt(question="Q?", context="C")

    assert "используя только приведённый ниже контекст" in prompt
    assert prompt.endswith("Ответ только на языке вопроса:")


def test_unknown_prompt_version_is_rejected(monkeypatch):
    import pytest

    monkeypatch.setenv("QA_PROMPT", "v9")
    with pytest.raises(ValueError, match="QA_PROMPT"):
        build_qa_prompt(question="Q?", context="C")


def test_answer_question_with_context_calls_llm_client_generate():
    llm_client = DummyLLMClient()
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="The project parses files.",
                metadata={"source_file": "README.md"},
            ),
            score=1.0,
        )
    ]

    answer = answer_question_with_context(
        question="What does the project do?",
        results=results,
        llm_client=llm_client,
    )

    assert answer == "Generated answer"
    assert len(llm_client.prompts) == 1
    assert "What does the project do?" in llm_client.prompts[0]
    assert "The project parses files." in llm_client.prompts[0]


def test_answer_question_with_context_skips_llm_when_results_empty():
    llm_client = DummyLLMClient()

    answer = answer_question_with_context(
        question="What does the project do?",
        results=[],
        llm_client=llm_client,
    )

    assert answer == NO_CONTEXT_MESSAGE
    assert llm_client.prompts == []
