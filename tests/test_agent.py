import json

import pytest

from file_agent.agent.agent import (
    FINAL_ANSWER_DEMAND,
    FORMAT_REMINDER,
    AgentResponse,
    FileAgent,
    answer_with_agent,
    build_system_prompt,
)
from file_agent.agent.tools import Tool, ToolError, ToolResult
from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult


class ScriptedLLM:
    """Chat client that replays a fixed list of replies."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(message) for message in messages])
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        return self.replies.pop(0)


def make_search_tool(results: list[SearchResult] | None = None, output: str = "Found passage"):
    calls: list[dict] = []

    def run(**kwargs) -> ToolResult:
        calls.append(kwargs)
        return ToolResult(output=output, sources=results or [])

    tool = Tool(
        name="search_documents",
        description="Search the documents.",
        parameters={"query": "string, required"},
        run=run,
    )
    return tool, calls


def make_result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, text=f"text {chunk_id}", metadata={"source_file": "doc.md"}),
        score=1.0,
    )


def action(tool: str, **arguments) -> str:
    return f"Action: {json.dumps({'tool': tool, 'arguments': arguments})}"


def test_agent_calls_tool_and_returns_final_answer():
    tool, calls = make_search_tool(results=[make_result("chunk-1")])
    llm = ScriptedLLM(
        [
            "Thought: I need to search.\n" + action("search_documents", query="report date"),
            "Thought: I have enough.\nFinal Answer: The report is from 2024.",
        ]
    )
    agent = FileAgent(llm_client=llm, tools=[tool])

    response = agent.run("When is the report from?")

    assert isinstance(response, AgentResponse)
    assert response.answer == "The report is from 2024."
    assert [source.chunk.id for source in response.sources] == ["chunk-1"]
    assert calls == [{"query": "report date"}]
    assert len(response.steps) == 2
    assert response.steps[0].tool == "search_documents"
    assert response.steps[0].observation == "Found passage"
    assert response.steps[1].thought == "I have enough."

    # The tool observation is fed back to the model on the next turn.
    final_messages = llm.calls[-1]
    assert final_messages[0]["role"] == "system"
    assert final_messages[-1] == {"role": "user", "content": "Observation: Found passage"}


def test_agent_strips_reasoning_blocks_before_parsing():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            "<think>Let me plan the search first.</think>\n"
            "Thought: search it\n" + action("search_documents", query="x"),
            "<think>done</think>Final Answer: Answer text.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.answer == "Answer text."
    assert calls == [{"query": "x"}]
    assert "<think>" not in response.steps[0].response


def test_agent_accepts_bare_json_tool_call_without_action_marker():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            '{"tool": "search_documents", "arguments": {"query": "totals"}}',
            "Final Answer: Done.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.answer == "Done."
    assert calls == [{"query": "totals"}]


def test_agent_treats_plain_text_reply_as_final_answer():
    tool, _ = make_search_tool()
    llm = ScriptedLLM(["The documents do not mention this."])

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.answer == "The documents do not mention this."
    assert len(response.steps) == 1


def test_agent_reminds_about_format_when_reply_is_unparseable():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            "Action: {broken json",
            "Final Answer: Recovered.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.answer == "Recovered."
    assert calls == []
    assert response.steps[0].observation == FORMAT_REMINDER


def test_agent_reports_unknown_tool_to_the_model():
    tool, _ = make_search_tool()
    llm = ScriptedLLM(
        [
            action("delete_documents"),
            "Final Answer: Understood.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    observation = response.steps[0].observation
    assert "Unknown tool 'delete_documents'" in observation
    assert "search_documents" in observation


def test_agent_feeds_tool_errors_back_as_observations():
    def run(**kwargs) -> ToolResult:
        raise ToolError("'query' is required.")

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    llm = ScriptedLLM(
        [
            action("search_documents"),
            "Final Answer: OK.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.steps[0].observation == "'query' is required."
    assert response.answer == "OK."


def test_agent_survives_unexpected_tool_crash():
    def run(**kwargs) -> ToolResult:
        raise RuntimeError("boom")

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    llm = ScriptedLLM(
        [
            action("search_documents", query="x"),
            "Final Answer: OK.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert "failed: boom" in response.steps[0].observation
    assert response.answer == "OK."


def test_agent_forces_final_answer_when_step_budget_is_exhausted():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            action("search_documents", query="first"),
            action("search_documents", query="second"),
            "Final Answer: Forced answer.",
        ]
    )
    agent = FileAgent(llm_client=llm, tools=[tool], max_steps=2)

    response = agent.run("Question?")

    assert response.answer == "Forced answer."
    assert len(calls) == 2
    # The last exchange demands an answer instead of allowing more tools.
    assert llm.calls[-1][-1] == {"role": "user", "content": FINAL_ANSWER_DEMAND}


def test_agent_deduplicates_sources_across_steps():
    first = make_result("chunk-1")
    duplicate = make_result("chunk-1")
    second = make_result("chunk-2")

    replies = iter([[first], [duplicate, second]])

    def run(**kwargs) -> ToolResult:
        return ToolResult(output="ok", sources=next(replies))

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    llm = ScriptedLLM(
        [
            action("search_documents", query="a"),
            action("search_documents", query="b"),
            "Final Answer: Done.",
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert [source.chunk.id for source in response.sources] == ["chunk-1", "chunk-2"]


def test_agent_prefers_first_marker_when_both_are_present():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            "Thought: t\nFinal Answer: Early answer.\n" + action("search_documents", query="x"),
        ]
    )

    response = FileAgent(llm_client=llm, tools=[tool]).run("Question?")

    assert response.answer.startswith("Early answer.")
    assert calls == []


def test_agent_requires_question_and_valid_configuration():
    tool, _ = make_search_tool()

    with pytest.raises(ValueError, match="question"):
        FileAgent(llm_client=ScriptedLLM([]), tools=[tool]).run("   ")
    with pytest.raises(ValueError, match="max_steps"):
        FileAgent(llm_client=ScriptedLLM([]), tools=[tool], max_steps=0)
    with pytest.raises(ValueError, match="tool"):
        FileAgent(llm_client=ScriptedLLM([]), tools=[])


def test_system_prompt_lists_every_tool():
    tool, _ = make_search_tool()
    other = Tool(name="list_documents", description="List files.", parameters={}, run=tool.run)

    prompt = build_system_prompt([tool, other])

    assert "search_documents" in prompt
    assert "list_documents" in prompt
    assert "Final Answer" in prompt


def test_answer_with_agent_builds_default_tools_over_retriever():
    class FakeRetriever:
        def __init__(self):
            self.search_calls = []

        def index(self, chunks):
            pass

        def search(self, query: str, top_k: int = 5):
            self.search_calls.append((query, top_k))
            return [make_result("chunk-9")]

        def clear(self):
            pass

    retriever = FakeRetriever()
    llm = ScriptedLLM(
        [
            action("search_documents", query="totals", top_k=2),
            "Final Answer: Answer from tools.",
        ]
    )

    response = answer_with_agent(
        question="Question?",
        llm_client=llm,
        retriever=retriever,
        documents=[],
    )

    assert response.answer == "Answer from tools."
    assert retriever.search_calls == [("totals", 2)]
    assert [source.chunk.id for source in response.sources] == ["chunk-9"]
