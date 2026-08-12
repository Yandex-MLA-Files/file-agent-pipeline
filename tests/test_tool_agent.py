import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    ToolAgentContext,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.rag_graph import build_tool_agent_graph
from file_agent.retrieval import SearchResult


class FakeToolCallingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt: str) -> str:
        raise AssertionError("generate must not be used by the tool agent")

    def chat_with_tools(self, messages, tools):
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected tool-calling LLM invocation")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.search_calls = []

    def index(self, chunks):
        self.results = [SearchResult(chunk=chunk, score=1.0) for chunk in chunks]

    def search(self, query: str, top_k: int = 5, source_file: str | None = None):
        self.search_calls.append((query, top_k, source_file))
        results = self.results
        if source_file is not None:
            results = [
                result
                for result in results
                if result.chunk.metadata.get("source_file") == source_file
            ]
        return results[:top_k]

    def clear(self):
        self.results = []


def make_document() -> Document:
    return Document(
        file_name="plan.md",
        file_type=".md",
        blocks=[
            Block(
                id="block-1",
                text="The project deadline is Friday.",
                type="text",
                metadata={"source_file": "plan.md"},
            )
        ],
        metadata={
            "table_of_contents": [{"title": "Schedule", "level": 1, "block_id": "heading-1"}]
        },
    )


def make_search_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id="chunk-1",
            text="deadline fragment",
            metadata={
                "context": "The project deadline is Friday.",
                "source_file": "plan.md",
                "page_number": 2,
            },
        ),
        score=0.9,
    )


def invoke_tool_agent(llm_client, retriever, max_tool_rounds=4, documents=None):
    return build_tool_agent_graph().invoke(
        {
            "question": "When is the deadline?",
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=documents or [make_document()],
            max_tool_rounds=max_tool_rounds,
        ),
    )


def tool_messages_seen_by_model(llm_client):
    return [
        message for message in llm_client.calls[-1]["messages"] if isinstance(message, ToolMessage)
    ]


@pytest.mark.parametrize("document_tool", DOCUMENT_TOOLS)
def test_document_tool_schemas_hide_injected_runtime(document_tool):
    schema = document_tool.tool_call_schema.model_json_schema()

    assert "runtime" not in schema["properties"]


def test_search_tool_exposes_filter_schema():
    schema = search_documents.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {"query", "top_k", "source_file"}


def test_tool_agent_searches_documents_and_returns_sources():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline", "top_k": 3},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The project deadline is Friday [plan.md, page 2]."),
        ]
    )
    retriever = FakeRetriever([make_search_result()])

    state = invoke_tool_agent(llm_client, retriever)

    response = state["response"]
    assert response.answer == "The project deadline is Friday [plan.md, page 2]."
    assert response.sources == [make_search_result()]
    assert response.search_queries == ["project deadline"]
    assert response.stop_reason == "tool_agent_completed"
    assert response.tool_calls == [
        {
            "name": "search_documents",
            "arguments": {"query": "project deadline", "top_k": 3},
        }
    ]
    assert retriever.search_calls == [("project deadline", 3, None)]
    assert tool_messages_seen_by_model(llm_client)
    assert state["messages"] == []
    assert llm_client.responses == []


def test_tool_agent_can_list_documents_without_retrieval():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_documents",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The indexed document is plan.md."),
        ]
    )
    retriever = FakeRetriever()

    state = invoke_tool_agent(llm_client, retriever)

    tool_messages = tool_messages_seen_by_model(llm_client)
    assert "plan.md" in str(tool_messages[0].content)
    assert state["response"].sources == []
    assert retriever.search_calls == []


def test_tool_agent_forces_final_answer_after_tool_round_limit():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline", "top_k": 5},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The available context says the deadline is Friday."),
        ]
    )
    retriever = FakeRetriever([make_search_result()])

    state = invoke_tool_agent(llm_client, retriever, max_tool_rounds=1)

    assert state["response"].answer.endswith("deadline is Friday.")
    assert llm_client.calls[0]["tool_names"] == [tool.name for tool in DOCUMENT_TOOLS]
    assert llm_client.calls[1]["tool_names"] == []
    assert "tool-call limit" in str(llm_client.calls[1]["messages"][0].content)
    assert retriever.search_calls == [("project deadline", 5, None)]


def test_tool_agent_returns_tool_errors_to_model():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "missing.pdf"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The requested document is not indexed."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever())

    tool_messages = tool_messages_seen_by_model(llm_client)
    assert "document is not indexed" in str(tool_messages[0].content)
    assert state["response"].answer == "The requested document is not indexed."


def test_tool_agent_can_filter_search_by_source_file():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {
                            "query": "project deadline",
                            "source_file": "PLAN.MD",
                            "top_k": 3,
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deadline is Friday."),
        ]
    )
    retriever = FakeRetriever(
        [
            make_search_result(),
            SearchResult(
                chunk=Chunk(
                    id="other-chunk",
                    text="Other deadline",
                    metadata={"source_file": "other.md"},
                ),
                score=0.8,
            ),
        ]
    )

    state = invoke_tool_agent(llm_client, retriever)

    assert retriever.search_calls == [("project deadline", 3, "plan.md")]
    assert state["response"].sources == [make_search_result()]
    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert payload["source_file"] == "plan.md"
    assert payload["results_count"] == 1


def test_tool_agent_can_read_full_context_after_search():
    long_context = "Beginning " + "x" * 3700 + " complete ending"
    result = SearchResult(
        chunk=Chunk(
            id="long-chunk",
            text="short match",
            metadata={"context": long_context, "source_file": "plan.md"},
        ),
        score=0.9,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "complete ending"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source_context",
                        "args": {"chunk_id": "long-chunk"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The context has a complete ending."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([result]))

    tool_messages = tool_messages_seen_by_model(llm_client)
    search_payload = json.loads(str(tool_messages[0].content))
    context_payload = json.loads(str(tool_messages[1].content))
    assert search_payload["results"][0]["text"].endswith("...")
    assert context_payload["text"] == long_context
    assert state["response"].sources == [result]


def make_structured_document() -> Document:
    return Document(
        file_name="handbook.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="heading-main",
                text="Main section",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=1,
            ),
            Block(
                id="main-body",
                text="Main section body.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="heading-child",
                text="Child section",
                type="heading",
                metadata={"hierarchy_level": 2},
                block_type=BlockType.HEADING,
                page_number=2,
            ),
            Block(
                id="child-body",
                text="Child section body.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=2,
            ),
            Block(
                id="table-1",
                text="| Name | Value |\n| --- | --- |\n| Accuracy | 0.9 |",
                type="table",
                block_type=BlockType.TABLE,
                page_number=2,
            ),
            Block(
                id="heading-next",
                text="Next section",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=3,
            ),
            Block(
                id="next-body",
                text="This text must not be returned with the main section.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=3,
            ),
        ],
    )


def test_tool_agent_can_navigate_outline_and_read_section():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "handbook.pdf"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_document_section",
                        "args": {
                            "source_file": "handbook.pdf",
                            "section_id": "heading-main",
                        },
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The main section includes its child section."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[make_structured_document()],
    )

    tool_messages = tool_messages_seen_by_model(llm_client)
    outline = json.loads(str(tool_messages[0].content))
    section = json.loads(str(tool_messages[1].content))
    assert outline["table_of_contents"][0]["section_id"] == "heading-main"
    assert outline["tables"][0]["table_id"] == "table-1"
    assert "Main section body." in section["text"]
    assert "Child section body." in section["text"]
    assert "must not be returned" not in section["text"]
    assert state["response"].sources[0].chunk.metadata["page_numbers"] == [1, 2]


def _invoke_single_read_tool(tool_name, arguments, documents):
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": arguments,
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Answer based on the selected location."),
        ]
    )
    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=documents,
    )
    return state, llm_client


@pytest.mark.parametrize(
    ("documents", "arguments", "expected_text", "metadata_key", "metadata_value"),
    [
        (
            [make_structured_document()],
            {"source_file": "handbook.pdf", "page_number": 3},
            "Next section",
            "page_number",
            3,
        ),
        (
            [
                Document(
                    file_name="deck.pptx",
                    file_type="pptx",
                    blocks=[
                        Block(
                            id="slide-2",
                            text="Slide 2 metrics",
                            type="pptx_slide",
                            metadata={"slide_number": 2},
                        )
                    ],
                )
            ],
            {"source_file": "deck.pptx", "slide_number": 2},
            "Slide 2 metrics",
            "slide_number",
            2,
        ),
        (
            [
                Document(
                    file_name="budget.xlsx",
                    file_type="xlsx",
                    blocks=[
                        Block(
                            id="sheet-1",
                            text="Month\tAmount\nJanuary\t100",
                            type="xlsx_sheet",
                            metadata={"sheet_name": "Budget"},
                            block_type=BlockType.TABLE,
                        )
                    ],
                )
            ],
            {"source_file": "budget.xlsx", "sheet_name": "budget"},
            "January",
            "sheet_name",
            "Budget",
        ),
    ],
)
def test_tool_agent_can_read_document_locations(
    documents,
    arguments,
    expected_text,
    metadata_key,
    metadata_value,
):
    state, llm_client = _invoke_single_read_tool("read_document_location", arguments, documents)

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert expected_text in payload["text"]
    assert payload["location"][metadata_key] == metadata_value
    assert state["response"].sources[0].chunk.metadata[metadata_key] == metadata_value


def test_tool_agent_can_read_table_rows_with_pagination():
    document = Document(
        file_name="budget.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text="Name\tValue\nA\t1\nB\t2\nC\t3",
                type="xlsx_sheet",
                metadata={"sheet_name": "Budget"},
                block_type=BlockType.TABLE,
            )
        ],
    )

    state, llm_client = _invoke_single_read_tool(
        "read_table",
        {
            "source_file": "budget.xlsx",
            "table_id": "sheet-1",
            "offset": 1,
            "limit": 2,
        },
        [document],
    )

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert payload["format"] == "tsv"
    assert payload["rows"] == ["A\t1", "B\t2"]
    assert payload["next_offset"] == 3
    assert state["response"].sources[0].chunk.metadata["table_id"] == "sheet-1"


def invoke_persistent_turn(
    graph,
    llm_client,
    retriever,
    question,
    thread_id="conversation-1",
    max_history_turns=6,
):
    return graph.invoke(
        {
            "question": question,
            "documents_count": 1,
            "chunks_count": 1,
        },
        config={"configurable": {"thread_id": thread_id}},
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=[make_document()],
            max_history_turns=max_history_turns,
        ),
    )


def test_persistent_agent_uses_compact_history_and_retrieves_again_each_turn():
    first_result = make_search_result()
    second_result = SearchResult(
        chunk=Chunk(
            id="quarter-2",
            text="Second-quarter revenue was 150 million rubles.",
            metadata={"source_file": "plan.md", "page_number": 3},
        ),
        score=0.95,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "first-quarter revenue"},
                        "id": "turn-1-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="First-quarter revenue was 120 million rubles."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "second-quarter revenue"},
                        "id": "turn-2-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Second-quarter revenue was 150 million rubles."),
        ]
    )
    retriever = FakeRetriever([first_result])
    checkpointer = InMemorySaver()
    graph = build_tool_agent_graph(checkpointer=checkpointer)

    first_state = invoke_persistent_turn(
        graph,
        llm_client,
        retriever,
        "What was the first-quarter revenue?",
    )
    retriever.results = [second_result]
    second_state = invoke_persistent_turn(
        graph,
        llm_client,
        retriever,
        "And in the second?",
    )

    second_turn_input = llm_client.calls[2]["messages"]
    assert [message.content for message in second_turn_input[1:]] == [
        "What was the first-quarter revenue?",
        "First-quarter revenue was 120 million rubles.",
        "And in the second?",
    ]
    assert not any(isinstance(message, ToolMessage) for message in second_turn_input)
    assert not any(
        isinstance(message, AIMessage) and message.tool_calls for message in second_turn_input
    )
    assert retriever.search_calls == [
        ("first-quarter revenue", 5, None),
        ("second-quarter revenue", 5, None),
    ]
    assert first_state["sources"] == [first_result]
    assert second_state["sources"] == [second_result]
    assert second_state["response"].sources == [second_result]
    assert second_state["messages"] == []

    persisted = graph.get_state({"configurable": {"thread_id": "conversation-1"}}).values
    assert persisted["messages"] == []
    assert [message.content for message in persisted["conversation_history"]] == [
        "What was the first-quarter revenue?",
        "First-quarter revenue was 120 million rubles.",
        "And in the second?",
        "Second-quarter revenue was 150 million rubles.",
    ]


def test_persistent_agent_limits_history_to_complete_recent_pairs():
    llm_client = FakeToolCallingLLM([AIMessage(content=f"Answer {turn}") for turn in range(1, 5)])
    graph = build_tool_agent_graph(checkpointer=InMemorySaver())

    for turn in range(1, 5):
        state = invoke_persistent_turn(
            graph,
            llm_client,
            FakeRetriever(),
            f"Question {turn}",
            thread_id="bounded-history",
            max_history_turns=2,
        )

    fourth_turn_input = llm_client.calls[3]["messages"]
    assert [message.content for message in fourth_turn_input[1:]] == [
        "Question 2",
        "Answer 2",
        "Question 3",
        "Answer 3",
        "Question 4",
    ]
    assert [message.content for message in state["conversation_history"]] == [
        "Question 3",
        "Answer 3",
        "Question 4",
        "Answer 4",
    ]
    assert state["messages"] == []


def test_different_thread_id_starts_without_previous_conversation():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(content="First thread answer"),
            AIMessage(content="New thread answer"),
        ]
    )
    graph = build_tool_agent_graph(checkpointer=InMemorySaver())

    invoke_persistent_turn(
        graph,
        llm_client,
        FakeRetriever(),
        "Question in the first chat",
        thread_id="thread-one",
    )
    state = invoke_persistent_turn(
        graph,
        llm_client,
        FakeRetriever(),
        "Question in a new chat",
        thread_id="thread-two",
    )

    assert [message.content for message in llm_client.calls[1]["messages"][1:]] == [
        "Question in a new chat"
    ]
    assert [message.content for message in state["conversation_history"]] == [
        "Question in a new chat",
        "New thread answer",
    ]


def test_grounding_prompt_requires_fresh_document_evidence_for_followups():
    first_model_messages = FakeToolCallingLLM([AIMessage(content="Answer")])

    invoke_tool_agent(first_model_messages, FakeRetriever())

    system_prompt = str(first_model_messages.calls[0]["messages"][0].content)
    assert "Conversation history" in system_prompt
    assert "It is not evidence" in system_prompt
    assert "For every new user turn" in system_prompt
