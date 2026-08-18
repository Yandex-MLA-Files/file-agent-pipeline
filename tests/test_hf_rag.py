import json

import pytest

from file_agent.chunking import Chunk
from file_agent.hf_dataset import QADatasetRecord
from file_agent.hf_rag import (
    process_hf_qa_record,
    process_qa_record,
    serialize_search_results,
)
from file_agent.llm.base import ToolCall, ToolCallResponse
from file_agent.rag import load_documents
from file_agent.retrieval import SearchResult


class ScriptedToolLLM:
    """Replays one ToolCallResponse per generate_with_tools() call, in order."""

    model = "fake/model"

    def __init__(self, responses: list[ToolCallResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, prompt: str) -> str:
        raise AssertionError("the ReAct agent must use generate_with_tools, not generate")

    def generate_with_tools(self, messages, tools, tool_choice="auto") -> ToolCallResponse:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        return self.responses.pop(0)


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise AssertionError("unused")

    def generate_with_tools(self, messages, tools, tool_choice="auto"):
        raise RuntimeError("Generation failed")


def tool_call(name: str, arguments: dict, call_id: str = "call-1") -> ToolCallResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return ToolCallResponse(content=None, tool_calls=[call])


def final_answer(text: str) -> ToolCallResponse:
    return ToolCallResponse(content=text, tool_calls=[])


def search_then_answer(query: str, answer: str, top_k: int | None = None) -> list[ToolCallResponse]:
    arguments = {"query": query} if top_k is None else {"query": query, "top_k": top_k}
    return [tool_call("search_documents", arguments), final_answer(answer)]


class FakeRetriever:
    def __init__(self):
        self.chunks = []
        self.index_calls = 0
        self.search_calls = []
        self.clear_calls = 0

    def index(self, chunks):
        self.index_calls += 1
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return [
            SearchResult(chunk=chunk, score=1.0 / rank)
            for rank, chunk in enumerate(self.chunks[:top_k], start=1)
        ]

    def clear(self):
        self.clear_calls += 1


def make_record():
    return QADatasetRecord(
        id="q0001",
        question="Which contexts were found?",
        answer="Gold answer",
        doc_ids=("q0001/first.txt", "q0001/second.txt"),
    )


def create_text_documents(tmp_path):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("First retrieved context", encoding="utf-8")
    second_path.write_text("Second retrieved context", encoding="utf-8")
    return [first_path, second_path]


def test_process_qa_record_generates_answer_and_serializes_exact_contexts(tmp_path):
    record = make_record()
    document_paths = create_text_documents(tmp_path)
    # search_then_answer's turns plus one more: the verification call that
    # fires whenever sources were gathered (see run_react_agent's faithfulness
    # gate), here echoing the same answer back unchanged.
    llm_client = ScriptedToolLLM(
        [*search_then_answer(record.question, "Generated answer"), final_answer("Generated answer")]
    )
    retriever = FakeRetriever()

    result = process_qa_record(
        record=record,
        document_paths=document_paths,
        llm_client=llm_client,
        top_k=2,
        retriever=retriever,
    )

    assert result.id == "q0001"
    assert result.question == record.question
    assert result.doc_ids == record.doc_ids
    assert result.answer_model == "Generated answer"
    assert result.answer == "Gold answer"
    assert retriever.index_calls == 1
    assert retriever.search_calls == [(record.question, 2)]
    assert retriever.clear_calls == 1
    assert len(llm_client.calls) == 3  # tool-call turn + final-answer turn + verification turn

    first_context, second_context = result.contexts
    assert first_context.rank == 1
    assert first_context.document_id == "q0001/first.txt"
    assert first_context.text == "First retrieved context"
    assert first_context.retrieval_text == "First retrieved context"
    assert first_context.score == 1.0
    assert json.loads(first_context.metadata_json)["dataset_record_id"] == "q0001"
    assert second_context.rank == 2
    assert second_context.document_id == "q0001/second.txt"
    assert second_context.text == "Second retrieved context"
    assert second_context.retrieval_text == "Second retrieved context"
    assert second_context.score == 0.5


def test_serialize_search_results_matches_small_to_big_llm_context():
    parent = "Complete parent section shown to the LLM"
    results = [
        SearchResult(
            chunk=Chunk(
                id="chunk-1",
                text="First retrieval fragment",
                metadata={
                    "context": parent,
                    "dataset_doc_id": "doc-1.pdf",
                    "page_number": 1,
                },
            ),
            score=1.0,
        ),
        SearchResult(
            chunk=Chunk(
                id="chunk-2",
                text="Second retrieval fragment",
                metadata={
                    "context": parent,
                    "dataset_doc_id": "doc-1.pdf",
                    "page_number": 1,
                },
            ),
            score=0.8,
        ),
        SearchResult(
            chunk=Chunk(
                id="chunk-3",
                text="Standalone passage",
                metadata={"dataset_doc_id": "doc-2.pdf", "page_number": 2},
            ),
            score=0.5,
        ),
    ]

    contexts = serialize_search_results(results)

    assert [context.rank for context in contexts] == [1, 2]
    assert contexts[0].text == parent
    assert contexts[0].retrieval_text == "First retrieval fragment"
    assert contexts[1].text == "Standalone passage"
    assert contexts[1].retrieval_text == "Standalone passage"
    assert "context" not in json.loads(contexts[0].metadata_json)


def test_generated_record_converts_to_output_dictionary(tmp_path):
    result = process_qa_record(
        record=make_record(),
        document_paths=create_text_documents(tmp_path),
        llm_client=ScriptedToolLLM([final_answer("Generated answer")]),
        top_k=1,
        retriever=FakeRetriever(),
    )

    output = result.to_dict()

    assert list(output) == [
        "id",
        "question",
        "doc_ids",
        "answer_model",
        "contexts",
        "answer",
    ]
    assert output["doc_ids"] == ["q0001/first.txt", "q0001/second.txt"]
    assert output["contexts"] == []


def test_process_hf_qa_record_downloads_documents_before_processing(monkeypatch, tmp_path):
    record = make_record()
    document_paths = create_text_documents(tmp_path)
    calls = []

    def fake_download_record_documents(**kwargs):
        calls.append(kwargs)
        return document_paths

    monkeypatch.setattr(
        "file_agent.hf_rag.download_record_documents",
        fake_download_record_documents,
    )

    result = process_hf_qa_record(
        record=record,
        dataset_id="owner/rag-qa",
        llm_client=ScriptedToolLLM([final_answer("Generated answer")]),
        revision="commit-sha",
        cache_dir=tmp_path / "cache",
        token="test-token",
        retriever=FakeRetriever(),
    )

    assert result.id == record.id
    assert calls == [
        {
            "record": record,
            "dataset_id": "owner/rag-qa",
            "revision": "commit-sha",
            "cache_dir": tmp_path / "cache",
            "token": "test-token",
        }
    ]


def test_process_qa_record_rejects_mismatched_document_paths(tmp_path):
    document_paths = create_text_documents(tmp_path)

    with pytest.raises(ValueError, match="document_paths count"):
        process_qa_record(
            record=make_record(),
            document_paths=document_paths[:1],
            llm_client=ScriptedToolLLM([]),
            retriever=FakeRetriever(),
        )


def test_process_qa_record_uses_supplied_document_loader(tmp_path):
    record = make_record()
    document_paths = create_text_documents(tmp_path)
    loader_calls = []

    def recording_loader(paths):
        loader_calls.append(paths)
        return load_documents(paths)

    result = process_qa_record(
        record=record,
        document_paths=document_paths,
        llm_client=ScriptedToolLLM([final_answer("Generated answer")]),
        retriever=FakeRetriever(),
        document_loader=recording_loader,
    )

    assert result.id == record.id
    assert loader_calls == [document_paths]


def test_process_qa_record_reports_no_matching_passages_to_the_llm(tmp_path):
    class EmptyRetriever(FakeRetriever):
        def search(self, query: str, top_k: int = 5):
            self.search_calls.append((query, top_k))
            return []

    llm_client = ScriptedToolLLM(
        [
            tool_call("search_documents", {"query": "Which contexts were found?"}),
            final_answer("No relevant information was found."),
        ]
    )
    retriever = EmptyRetriever()

    result = process_qa_record(
        record=make_record(),
        document_paths=create_text_documents(tmp_path),
        llm_client=llm_client,
        retriever=retriever,
    )

    assert result.answer_model == "No relevant information was found."
    assert result.contexts == ()
    assert retriever.clear_calls == 1
    # The tool observation fed back to the LLM confirms no passages, not a crash.
    tool_message = llm_client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "No matching passages found."


def test_process_qa_record_clears_retriever_when_generation_fails(tmp_path):
    retriever = FakeRetriever()

    with pytest.raises(RuntimeError, match="Generation failed"):
        process_qa_record(
            record=make_record(),
            document_paths=create_text_documents(tmp_path),
            llm_client=FailingLLM(),
            retriever=retriever,
        )

    assert retriever.clear_calls == 1
