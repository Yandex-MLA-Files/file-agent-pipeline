from file_agent.rag import answer_files, answer_documents


class DummyLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


def test_answer_files_runs_full_rag_pipeline(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Project\n\nThe project parses markdown files.",
        encoding="utf-8",
    )
    llm_client = DummyLLM()

    response = answer_files(
        file_paths=[file_path],
        question="What parses markdown?",
        llm_client=llm_client,
        use_semantic=False,
    )

    assert response.answer == "Generated answer"
    assert response.documents_count == 1
    assert response.chunks_count == 1
    assert [source.chunk.id for source in response.sources] == ["block-1-chunk-1"]
    assert len(llm_client.prompts) == 1
    assert "The project parses markdown files." in llm_client.prompts[0]


def test_answer_documents_uses_no_context_message_without_matches():
    llm_client = DummyLLM()

    response = answer_documents(
        documents=[],
        question="What is the answer?",
        llm_client=llm_client,
        use_semantic=False,
    )

    assert "No relevant context" in response.answer
    assert response.sources == []
    assert response.documents_count == 0
    assert response.chunks_count == 0
    assert llm_client.prompts == []


def test_chunk_documents_combines_multiple_documents(tmp_path):
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_path.write_text("First document text", encoding="utf-8")
    second_path.write_text("Second document text", encoding="utf-8")

    response = answer_files(
        file_paths=[first_path, second_path],
        question="Second document",
        llm_client=DummyLLM(),
        use_semantic=False,
    )

    assert response.documents_count == 2
    assert response.chunks_count == 2
    assert response.sources[0].chunk.metadata["source_file"] == "second.md"
