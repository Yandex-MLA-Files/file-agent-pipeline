import pytest

from file_agent.agent.loop import SYSTEM_PROMPT, run_react_agent
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


def final_answer(text: str, usage: dict | None = None) -> ToolCallResponse:
    return ToolCallResponse(content=text, tool_calls=[], usage=usage)


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


def test_run_react_agent_exposes_the_last_calls_token_usage():
    usage = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}
    llm = ScriptedLLM([final_answer("The answer", usage=usage)])

    response = run_react_agent("question", llm, tools=[])

    assert response.last_usage == usage


def test_run_react_agent_dispatches_a_tool_call_and_collects_sources():
    result = make_search_result("relevant passage")
    llm = ScriptedLLM(
        [
            tool_call("search_documents", {"query": "q"}),
            final_answer("Based on the passage"),
            final_answer("Based on the passage"),  # verification call (sources present)
        ]
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


def test_run_react_agent_revises_the_answer_via_the_verification_call():
    result = make_search_result("relevant passage")
    llm = ScriptedLLM(
        [
            tool_call("search_documents", {"query": "q"}),
            final_answer("draft answer"),
            final_answer("verified answer"),
        ]
    )

    response = run_react_agent("question", llm, tools=[search_tool(result)])

    assert response.answer == "verified answer"
    assert len(llm.calls) == 3
    verification_call = llm.calls[-1]
    assert verification_call["tool_choice"] == "none"
    verification_prompt = verification_call["messages"][-1]["content"]
    assert "draft answer" in verification_prompt
    assert "relevant passage" in verification_prompt


def test_run_react_agent_skips_verification_when_disabled():
    # verify_answer=False is the CLI/batch default (latency tradeoff) -
    # sources present but the caller opted out, so no extra call should fire.
    result = make_search_result("relevant passage")
    llm = ScriptedLLM([tool_call("search_documents", {"query": "q"}), final_answer("draft answer")])

    response = run_react_agent("question", llm, tools=[search_tool(result)], verify_answer=False)

    assert response.answer == "draft answer"
    assert len(llm.calls) == 2


def test_run_react_agent_skips_verification_without_sources():
    # A purely conversational answer (no tool calls, nothing retrieved) has
    # no evidence to check against - the verification call must not fire.
    llm = ScriptedLLM([final_answer("The answer")])

    response = run_react_agent("question", llm, tools=[])

    assert response.answer == "The answer"
    assert len(llm.calls) == 1


def test_run_react_agent_keeps_the_draft_answer_when_verification_fails():
    # The verification call is a safety net, not a hard dependency - if it
    # breaks, the turn must still return the (unverified) draft, not crash.
    result = make_search_result("relevant passage")

    class FailingVerificationLLM(ScriptedLLM):
        def generate_with_tools(self, messages, tools, tool_choice="auto"):
            if tool_choice == "none":
                raise RuntimeError("verification call failed")
            return super().generate_with_tools(messages, tools, tool_choice)

    llm = FailingVerificationLLM(
        [tool_call("search_documents", {"query": "q"}), final_answer("draft answer")]
    )

    response = run_react_agent("question", llm, tools=[search_tool(result)])

    assert response.answer == "draft answer"


def test_run_react_agent_verifies_the_forced_final_answer_too():
    result = make_search_result("relevant passage")
    echo = Tool(
        name="echo",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=lambda **kwargs: ToolResult(content="found", sources=[result]),
    )
    llm = ScriptedLLM(
        [
            tool_call("echo", {}),
            tool_call("echo", {}),
            final_answer("forced draft"),
            final_answer("forced verified"),
        ]
    )

    response = run_react_agent("question", llm, tools=[echo], max_iterations=2)

    assert response.answer == "forced verified"
    assert llm.calls[-2]["tool_choice"] == "none"  # the forced-answer call
    assert llm.calls[-1]["tool_choice"] == "none"  # the verification call


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
    forced_usage = {"prompt_tokens": 999, "completion_tokens": 12, "total_tokens": 1011}
    llm = ScriptedLLM(
        [
            tool_call("echo", {}),
            tool_call("echo", {}),
            final_answer("forced answer", usage=forced_usage),
        ]
    )

    response = run_react_agent("question", llm, tools=[echo], max_iterations=2)

    assert response.answer == "forced answer"
    assert response.iterations == 3
    assert llm.calls[-1]["tool_choice"] == "none"
    assert response.last_usage == forced_usage


def test_run_react_agent_stops_calling_tools_early_when_context_gets_full():
    # Regression for a real failure: 5 straight search_documents rounds grew
    # the prompt to ~32513/32768 tokens - so full that even the smallest
    # possible requested output still overflowed the model's context window
    # (a hard 400, not a graceful shorter answer). A tiny context_length here
    # makes one big tool result enough to trigger the same shape of problem.
    class ClientWithContextLimit(ScriptedLLM):
        context_length = 2000

    big_tool = Tool(
        name="big_tool",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=lambda **kwargs: ToolResult(content="X" * 3000),
    )
    llm = ClientWithContextLimit(
        [
            tool_call("big_tool", {}),
            final_answer("forced draft"),
        ]
    )

    response = run_react_agent("question", llm, tools=[big_tool], max_iterations=10)

    # Only 2 calls (one tool round + the forced final answer), not anywhere
    # near all 10 iterations - the context-size check broke out early.
    assert len(llm.calls) == 2
    assert llm.calls[-1]["tool_choice"] == "none"
    assert response.answer == "forced draft"
    assert response.iterations == 3


def test_run_react_agent_ignores_context_headroom_without_a_context_length():
    # ScriptedLLM has no context_length attribute at all (unlike a real
    # OpenAILLMClient) - the check must no-op, not crash on getattr.
    echo = Tool(
        name="echo",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=echo_tool,
    )
    llm = ScriptedLLM([tool_call("echo", {}), final_answer("answer")])

    response = run_react_agent("question", llm, tools=[echo])

    assert response.answer == "answer"


def test_run_react_agent_splices_conversation_history_between_system_and_question():
    llm = ScriptedLLM([final_answer("second answer")])
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    run_react_agent("second question", llm, tools=[], conversation_history=history)

    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "first question"}
    assert messages[2] == {"role": "assistant", "content": "first answer"}
    assert messages[3] == {"role": "user", "content": "second question"}
    assert len(messages) == 4


def test_run_react_agent_works_without_conversation_history():
    llm = ScriptedLLM([final_answer("answer")])

    run_react_agent("question", llm, tools=[], conversation_history=None)

    messages = llm.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "question"},
    ]


def test_run_react_agent_appends_history_summary_to_the_system_prompt():
    llm = ScriptedLLM([final_answer("answer")])

    run_react_agent(
        "question", llm, tools=[], history_summary="User previously asked about pricing."
    )

    system_message = llm.calls[0]["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"].startswith(SYSTEM_PROMPT)
    assert "User previously asked about pricing." in system_message["content"]


def test_run_react_agent_system_prompt_is_unchanged_without_a_history_summary():
    llm = ScriptedLLM([final_answer("answer")])

    run_react_agent("question", llm, tools=[], history_summary=None)

    assert llm.calls[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_run_react_agent_propagates_genuine_llm_failures():
    class FailingLLM:
        model = "fake/model"

        def generate(self, prompt: str) -> str:
            raise AssertionError("unused")

        def generate_with_tools(self, messages, tools, tool_choice="auto"):
            raise RuntimeError("LLM unreachable")

    with pytest.raises(RuntimeError, match="LLM unreachable"):
        run_react_agent("question", FailingLLM(), tools=[])
