import pytest

from file_agent.agent.loop import run_react_agent
from file_agent.agent.tools import Tool, ToolResult
from file_agent.chunking import Chunk
from file_agent.llm.base import ToolCall, ToolCallResponse
from file_agent.retrieval import SearchResult


class ScriptedLLM:
    model = "fake/model"

    def __init__(self, responses: list[ToolCallResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, prompt: str) -> str:
        raise AssertionError("unused")

    def generate_with_tools(self, messages, tools, tool_choice="auto") -> ToolCallResponse:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        return self.responses.pop(0)


def tool_call(name: str, arguments: dict, call_id: str = "call-1") -> ToolCallResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return ToolCallResponse(content=None, tool_calls=[call])


def final_answer(text: str) -> ToolCallResponse:
    return ToolCallResponse(content=text, tool_calls=[])


def make_search_result(text: str) -> SearchResult:
    return SearchResult(chunk=Chunk(id=text, text=text, metadata={}), score=1.0)


def echo_tool(**kwargs) -> ToolResult:
    return ToolResult(content=f"echoed: {kwargs}")


def failing_tool(**kwargs) -> ToolResult:
    raise RuntimeError("tool exploded")


def search_tool(result: SearchResult) -> Tool:
    return Tool(
        name="search_documents",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=lambda **kwargs: ToolResult(content=result.chunk.text, sources=[result]),
    )


def test_run_react_agent_returns_immediate_final_answer():
    llm = ScriptedLLM([final_answer("The answer")])

    response = run_react_agent("question", llm, tools=[])

    assert response.answer == "The answer"
    assert response.sources == []
    assert response.iterations == 1
    assert len(llm.calls) == 1


def test_run_react_agent_dispatches_a_tool_call_and_collects_sources():
    result = make_search_result("relevant passage")
    llm = ScriptedLLM(
        [tool_call("search_documents", {"query": "q"}), final_answer("Based on the passage")]
    )

    response = run_react_agent("question", llm, tools=[search_tool(result)])

    assert response.answer == "Based on the passage"
    assert response.sources == [result]
    assert response.iterations == 2

    # The tool observation was fed back into the conversation before the final call.
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "relevant passage",
    }


def test_run_react_agent_reports_unknown_tool_without_crashing():
    llm = ScriptedLLM([tool_call("does_not_exist", {}), final_answer("done")])

    response = run_react_agent("question", llm, tools=[])

    assert response.answer == "done"
    tool_message = llm.calls[1]["messages"][-1]
    assert "unknown tool" in tool_message["content"]


def test_run_react_agent_turns_tool_exceptions_into_observations():
    broken_tool = Tool(
        name="broken",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=failing_tool,
    )
    llm = ScriptedLLM([tool_call("broken", {}), final_answer("recovered")])

    response = run_react_agent("question", llm, tools=[broken_tool])

    assert response.answer == "recovered"
    tool_message = llm.calls[1]["messages"][-1]
    assert "tool exploded" in tool_message["content"]


def test_run_react_agent_forces_a_final_answer_at_the_iteration_cap():
    echo = Tool(
        name="echo",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=echo_tool,
    )
    # The agent keeps calling the tool forever; max_iterations=2 must still terminate.
    llm = ScriptedLLM(
        [
            tool_call("echo", {}),
            tool_call("echo", {}),
            final_answer("forced answer"),
        ]
    )

    response = run_react_agent("question", llm, tools=[echo], max_iterations=2)

    assert response.answer == "forced answer"
    assert response.iterations == 3
    assert llm.calls[-1]["tool_choice"] == "none"


def test_run_react_agent_propagates_genuine_llm_failures():
    class FailingLLM:
        model = "fake/model"

        def generate(self, prompt: str) -> str:
            raise AssertionError("unused")

        def generate_with_tools(self, messages, tools, tool_choice="auto"):
            raise RuntimeError("LLM unreachable")

    with pytest.raises(RuntimeError, match="LLM unreachable"):
        run_react_agent("question", FailingLLM(), tools=[])
