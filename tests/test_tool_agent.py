from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    TOOL_AGENT_SYSTEM_PROMPT,
    ToolAgentContext,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, Document
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

    def search(self, query: str, top_k: int = 5):
        self.search_calls.append((query, top_k))
        return self.results[:top_k]

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


def invoke_tool_agent(llm_client, retriever, max_tool_rounds=4):
    return build_tool_agent_graph().invoke(
        {
            "messages": [
                SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
                HumanMessage(content="When is the deadline?"),
            ],
            "question": "When is the deadline?",
            "sources": [],
            "search_queries": [],
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=[make_document()],
            max_tool_rounds=max_tool_rounds,
        ),
    )


def test_search_tool_schema_hides_injected_runtime():
    schema = search_documents.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {"query", "top_k"}
    assert "runtime" not in schema["properties"]


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
    assert retriever.search_calls == [("project deadline", 3)]
    assert any(isinstance(message, ToolMessage) for message in state["messages"])
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

    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
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
    assert retriever.search_calls == [("project deadline", 5)]


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

    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    assert "document is not indexed" in str(tool_messages[0].content)
    assert state["response"].answer == "The requested document is not indexed."
