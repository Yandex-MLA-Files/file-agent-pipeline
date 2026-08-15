import json

import fitz
import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    MAX_TOOL_CONTENT_LENGTH,
    ToolAgentContext,
    calculate,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.llm.base import EmptyLLMResponseError
from file_agent.rag_graph import build_tool_agent_graph
from file_agent.retrieval import SearchResult
from file_agent.vlm.base import VLMClient


class FakeToolCallingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt: str) -> str:
        raise AssertionError("generate must not be used by the tool agent")

    def chat_with_tools(self, messages, tools, tool_choice="auto"):
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
                "tool_choice": tool_choice,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected tool-calling LLM invocation")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


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


def invoke_tool_agent(
    llm_client,
    retriever,
    max_tool_rounds=4,
    documents=None,
    vlm_client=None,
    asset_store=None,
    require_evidence_tool=False,
):
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
            vlm_client=vlm_client,
            asset_store=asset_store,
            require_evidence_tool=require_evidence_tool,
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


def test_visual_tool_does_not_expose_paths_or_bounding_boxes():
    visual_tool = next(tool for tool in DOCUMENT_TOOLS if tool.name == "analyze_document_visual")
    schema = visual_tool.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {
        "source_file",
        "question",
        "visual_id",
        "page_number",
    }


def test_new_tool_schemas_are_small_and_explicit():
    read_tool = next(tool for tool in DOCUMENT_TOOLS if tool.name == "read_document")

    assert set(read_tool.tool_call_schema.model_json_schema()["properties"]) == {
        "source_file",
        "offset",
    }
    assert set(calculate.tool_call_schema.model_json_schema()["properties"]) == {
        "operation",
        "values",
    }


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
    assert response.sources[0].chunk.id == make_search_result().chunk.id
    assert response.sources[0].chunk.metadata["_llm_context"] == ("The project deadline is Friday.")
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


def test_tool_agent_recovers_an_empty_response_after_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            EmptyLLMResponseError("LLM returned an empty response"),
            AIMessage(content="The project deadline is Friday [plan.md, page 2]."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([make_search_result()]))

    assert state["response"].answer == "The project deadline is Friday [plan.md, page 2]."
    assert len(llm_client.calls) == 3
    assert llm_client.calls[-1]["tool_names"] == []
    assert "Return the final answer now" in llm_client.calls[-1]["messages"][-1].content


def test_tool_agent_can_require_a_document_evidence_tool():
    llm_client = FakeToolCallingLLM([AIMessage(content="An unsupported direct answer.")])

    with pytest.raises(ValueError, match="without successfully using"):
        invoke_tool_agent(
            llm_client,
            FakeRetriever(),
            require_evidence_tool=True,
        )

    assert llm_client.calls[0]["tool_choice"] == "required"


def test_required_document_evidence_returns_to_auto_after_a_tool_result():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The project deadline is Friday."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([make_search_result()]),
        require_evidence_tool=True,
    )

    assert state["response"].answer == "The project deadline is Friday."
    assert [call["tool_choice"] for call in llm_client.calls] == ["required", "auto"]


def test_required_evidence_allows_a_successful_search_with_no_matches():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "missing fact"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The documents do not contain enough information."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        require_evidence_tool=True,
    )

    assert state["response"].sources == []
    assert state["response"].search_queries == ["missing fact"]


def test_calculate_does_not_count_as_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"operation": "add", "values": [2, 2]},
                        "id": "calculate-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="4"),
        ]
    )

    with pytest.raises(ValueError, match="without successfully using"):
        invoke_tool_agent(
            llm_client,
            FakeRetriever(),
            require_evidence_tool=True,
        )

    assert [call["tool_choice"] for call in llm_client.calls] == ["required", "required"]


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
    final_messages = llm_client.calls[1]["messages"]
    assert isinstance(final_messages[0], SystemMessage)
    assert "uploaded documents" in str(final_messages[0].content)
    assert "tool-call limit" in str(final_messages[0].content)
    assert sum(isinstance(message, SystemMessage) for message in final_messages) == 1
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
    assert state["response"].sources[0].chunk.id == make_search_result().chunk.id
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
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "complete ending"},
                        "id": "call-3",
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
    repeated_search_payload = json.loads(str(tool_messages[2].content))
    assert search_payload["results"][0]["text"].endswith("...")
    assert context_payload["text"] == long_context
    assert repeated_search_payload["results"][0]["text"].endswith("...")
    assert state["response"].sources[0].chunk.id == result.chunk.id
    assert state["response"].sources[0].chunk.metadata["_llm_context"] == long_context


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


def test_direct_read_source_preserves_dataset_document_id():
    document = make_structured_document()
    for block in document.blocks:
        block.metadata["dataset_doc_id"] = "q0001/handbook.pdf"
        block.metadata["dataset_record_id"] = "q0001"

    state, _ = _invoke_single_read_tool(
        "read_document_location",
        {"source_file": "handbook.pdf", "page_number": 3},
        [document],
    )

    metadata = state["response"].sources[0].chunk.metadata
    assert metadata["dataset_doc_id"] == "q0001/handbook.pdf"
    assert metadata["dataset_record_id"] == "q0001"


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


def test_tool_agent_can_read_an_unstructured_document_with_pagination():
    text = "Start of document. " + "x" * MAX_TOOL_CONTENT_LENGTH + " End of document."
    document = Document(
        file_name="notes.txt",
        file_type="txt",
        blocks=[
            Block(
                id="block-1",
                text=text,
                type="plain_text",
                metadata={"dataset_doc_id": "q1/notes.txt"},
            )
        ],
    )

    state, llm_client = _invoke_single_read_tool(
        "read_document",
        {"source_file": "notes.txt"},
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["text"].startswith("Start of document.")
    assert len(payload["text"]) <= MAX_TOOL_CONTENT_LENGTH
    assert payload["next_offset"] is not None
    assert payload["total_characters"] == len(text)
    source = state["response"].sources[0]
    assert source.chunk.id == "document:notes.txt:0"
    assert source.chunk.metadata["evidence_type"] == "document_read"
    assert source.chunk.metadata["dataset_doc_id"] == "q1/notes.txt"

    _, next_llm_client = _invoke_single_read_tool(
        "read_document",
        {
            "source_file": "notes.txt",
            "offset": payload["next_offset"],
        },
        [document],
    )
    next_payload = json.loads(str(tool_messages_seen_by_model(next_llm_client)[0].content))
    assert next_payload["text"].endswith("End of document.")
    assert next_payload["next_offset"] is None


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


@pytest.mark.parametrize(
    ("arguments", "expected_result"),
    [
        (
            {
                "operation": "count_distinct",
                "value_column": "category",
            },
            {"count": 2, "values": ["X", "Y"]},
        ),
        (
            {
                "operation": "sum",
                "value_column": "stock",
                "multiply_by": "price",
            },
            {"value": "180"},
        ),
        (
            {
                "operation": "min",
                "value_column": "stock",
                "top_n": 1,
            },
            {
                "rows": [
                    {
                        "value": "1",
                        "row": {
                            "product": "C",
                            "stock": "1",
                            "price": "100",
                            "category": "Y",
                            "amount": "12",
                            "region": "North",
                        },
                    }
                ]
            },
        ),
        (
            {
                "operation": "group_sum",
                "value_column": "stock",
                "multiply_by": "price",
                "group_by": "category",
                "top_n": 2,
            },
            {
                "groups": [
                    {"group": "Y", "value": "100"},
                    {"group": "X", "value": "80"},
                ]
            },
        ),
    ],
)
def test_tool_agent_can_analyze_all_table_rows(arguments, expected_result):
    document = Document(
        file_name="inventory.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text=(
                    "product\tstock\tprice\tcategory\tamount\tregion\n"
                    "A\t2\t10\tX\t5\tNorth\n"
                    "B\t3\t20\tX\t8\tSouth\n"
                    "C\t1\t100\tY\t12\tNorth"
                ),
                type="xlsx_sheet",
                metadata={"sheet_name": "Inventory"},
                block_type=BlockType.TABLE,
            )
        ],
    )
    state, llm_client = _invoke_single_read_tool(
        "analyze_table",
        {
            "source_file": "inventory.xlsx",
            "table_id": "sheet-1",
            **arguments,
        },
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_data_rows"] == 3
    assert payload["rows_skipped"] == 0
    assert payload["result"] == expected_result
    assert state["response"].sources[0].chunk.metadata["table_operation"] == arguments["operation"]


@pytest.mark.parametrize(
    ("operation", "values", "expected_result", "expected_unit"),
    [
        ("add", ["100.1", "20.2"], "120.3", None),
        ("subtract", ["150", "120"], "30", None),
        ("multiply", ["12.5", "4"], "50", None),
        ("divide", ["150", "120"], "1.25", None),
        ("average", ["10", "20", "30"], "20", None),
        ("percentage_of", ["30", "120"], "25", "percent"),
        ("percent_change", ["120", "150"], "25", "percent"),
    ],
)
def test_calculate_performs_decimal_arithmetic(
    operation,
    values,
    expected_result,
    expected_unit,
):
    payload = json.loads(calculate.invoke({"operation": operation, "values": values}))

    assert payload == {
        "operation": operation,
        "values": values,
        "result": expected_result,
        "unit": expected_unit,
    }


@pytest.mark.parametrize(
    ("operation", "values", "error"),
    [
        ("divide", ["1", "0"], "divide by zero"),
        ("percent_change", ["0", "10"], "old value"),
        ("subtract", ["1"], "exactly two"),
        ("add", [], "at least one"),
    ],
)
def test_calculate_rejects_invalid_input(operation, values, error):
    with pytest.raises(ValueError, match=error):
        calculate.invoke({"operation": operation, "values": values})


def test_tool_agent_can_calculate_from_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "quarterly revenue"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {
                            "operation": "percent_change",
                            "values": [120, 150],
                        },
                        "id": "calculate-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Revenue increased by 25%."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([make_search_result()]),
        require_evidence_tool=True,
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[1].content))
    assert payload["result"] == "25"
    assert payload["unit"] == "percent"
    assert state["response"].answer == "Revenue increased by 25%."
    assert [call["name"] for call in state["response"].tool_calls] == [
        "search_documents",
        "calculate",
    ]
    assert state["response"].sources[0].chunk.id == make_search_result().chunk.id


class RecordingVLM(VLMClient):
    def __init__(self, answer="Revenue rises from Q1 to Q2."):
        self.answer = answer
        self.calls = []

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        self.calls.append({"size": image.size, "prompt": prompt})
        return self.answer


def make_visual_pdf() -> tuple[bytes, Document]:
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=300)
    page.insert_text((30, 30), "Quarterly revenue")
    page.draw_rect(fitz.Rect(80, 80, 300, 230), fill=(0, 0, 1))
    contents = pdf.tobytes()
    pdf.close()

    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="heading-revenue",
                text="Revenue",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=1,
            ),
            Block(
                id="text-revenue",
                text="Quarterly revenue chart for 2026.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="visual-revenue",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                bbox=(80.0, 80.0, 300.0, 230.0),
                vlm_description="A bar chart with quarterly revenue.",
            ),
        ],
    )
    document.build_table_of_contents()
    return contents, document


def test_tool_agent_discovers_and_analyzes_pdf_visual():
    pdf_bytes, document = make_visual_pdf()
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put(document.file_name, pdf_bytes)
    vlm_client = RecordingVLM()
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "report.pdf"},
                        "id": "outline-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "visual_id": "visual-revenue",
                            "question": "How did revenue change?",
                        },
                        "id": "visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Revenue rises from Q1 to Q2 [report.pdf, page 1]."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    tool_messages = tool_messages_seen_by_model(llm_client)
    outline = json.loads(str(tool_messages[0].content))
    analysis = json.loads(str(tool_messages[1].content))
    assert outline["visuals"][0]["visual_id"] == "visual-revenue"
    assert outline["visuals"][0]["description"] == "A bar chart with quarterly revenue."
    assert analysis["analysis"] == "Revenue rises from Q1 to Q2."
    assert analysis["page_number"] == 1
    assert analysis["visual_id"] == "visual-revenue"
    assert vlm_client.calls[0]["size"] == (488, 348)
    assert "How did revenue change?" in vlm_client.calls[0]["prompt"]
    assert "Quarterly revenue chart for 2026." in vlm_client.calls[0]["prompt"]
    assert state["response"].sources[0].chunk.id == "visual:report.pdf:visual-revenue"
    assert state["response"].sources[0].chunk.metadata["evidence_type"] == "visual_analysis"


def test_visual_tool_can_analyze_full_pdf_page_as_fallback():
    pdf_bytes, document = make_visual_pdf()
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put(document.file_name, pdf_bytes)
    vlm_client = RecordingVLM("The page contains a blue chart.")
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "page_number": 1,
                            "question": "What is visible on this page?",
                        },
                        "id": "page-visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The page contains a blue chart [report.pdf, page 1]."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["visual_id"] is None
    assert payload["page_number"] == 1
    assert vlm_client.calls[0]["size"] == (800, 600)
    assert state["response"].sources[0].chunk.id == "visual:report.pdf:page-1"


def test_visual_tool_returns_configuration_error_to_agent():
    _, document = make_visual_pdf()
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "page_number": 1,
                            "question": "Analyze the chart",
                        },
                        "id": "visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Visual analysis is unavailable."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
    )

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    assert "no VLM backend is configured" in str(tool_message.content)
    assert state["response"].sources == []


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
    assert [source.chunk.id for source in first_state["sources"]] == [first_result.chunk.id]
    assert [source.chunk.id for source in second_state["sources"]] == [second_result.chunk.id]
    assert [source.chunk.id for source in second_state["response"].sources] == [
        second_result.chunk.id
    ]
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
