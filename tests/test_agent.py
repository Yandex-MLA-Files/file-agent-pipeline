import json

import pytest

from file_agent.agent.agent import (
    DEFAULT_MAX_OBSERVATION_CHARS,
    FINAL_ANSWER_DEMAND,
    FORMAT_REMINDER,
    LAST_STEP_WARNING,
    NO_ANSWER_MESSAGE,
    REPEATED_CALL_NOTE,
    VERIFY_KEEP_TOKEN,
    AgentResponse,
    AgentSession,
    AgentSettings,
    FileAgent,
    _parse_reply,
    _split_citations,
    answer_with_agent,
    build_system_prompt,
)
from file_agent.agent.passages import PassageRegistry
from file_agent.agent.tools import Tool, ToolError, ToolResult
from file_agent.chunking import Chunk
from file_agent.retrieval import SearchResult

NO_VERIFY = AgentSettings(verify=False)


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


def make_result(chunk_id: str, text: str | None = None) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id, text=text or f"text {chunk_id}", metadata={"source_file": "doc.md"}
        ),
        score=1.0,
    )


def action(tool: str, **arguments) -> str:
    return f"Action: {json.dumps({'tool': tool, 'arguments': arguments})}"


def make_agent(llm, tools, **kwargs) -> FileAgent:
    kwargs.setdefault("settings", NO_VERIFY)
    return FileAgent(llm_client=llm, tools=tools, **kwargs)


def test_agent_calls_tool_and_returns_final_answer():
    tool, calls = make_search_tool(results=[make_result("chunk-1")])
    llm = ScriptedLLM(
        [
            "Thought: I need to search.\n" + action("search_documents", query="report date"),
            "Thought: I have enough.\nFinal Answer: The report is from 2024.",
        ]
    )
    agent = make_agent(llm, [tool])

    response = agent.run("When is the report from?")

    assert isinstance(response, AgentResponse)
    assert response.answer == "The report is from 2024."
    assert [source.chunk.id for source in response.sources] == ["chunk-1"]
    assert calls == [{"query": "report date"}]
    assert len(response.steps) == 2
    assert response.steps[0].tool == "search_documents"
    assert response.steps[0].observation == "Found passage"
    assert response.steps[1].thought == "I have enough."
    assert response.citations == []

    # The tool observation is fed back to the model on the next turn.
    final_messages = llm.calls[-1]
    assert final_messages[0]["role"] == "system"
    assert final_messages[-1] == {"role": "user", "content": "Observation: Found passage"}


def test_agent_exports_only_the_cited_passages_and_strips_the_sources_line():
    registry = PassageRegistry()
    first = registry.add_text("Revenue was 96 083.", "report.pdf", tool="search_documents")
    second = registry.add_text("Unrelated passage.", "report.pdf", tool="search_documents")
    third = registry.add_text("Costs were 70 000.", "report.pdf", tool="find_text")

    def run(**kwargs) -> ToolResult:
        return ToolResult(output="obs", passages=[first, second, third])

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    llm = ScriptedLLM(
        [
            action("search_documents", query="revenue"),
            "Thought: done.\nFinal Answer: Revenue was 96 083 and costs 70 000.\nSources: P3, P1",
        ]
    )

    response = make_agent(llm, [tool], registry=registry).run("Revenue and costs?")

    assert response.answer == "Revenue was 96 083 and costs 70 000."
    assert response.citations == ["P3", "P1"]
    assert [source.chunk.text for source in response.sources] == [
        "Costs were 70 000.",
        "Revenue was 96 083.",
    ]


def test_agent_exports_everything_seen_when_nothing_is_cited_or_citations_are_off():
    registry = PassageRegistry()
    passages = [registry.add_text(f"text {i}", "doc.md") for i in range(3)]

    def run(**kwargs) -> ToolResult:
        return ToolResult(output="obs", passages=passages)

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)

    uncited = make_agent(
        ScriptedLLM([action("search_documents"), "Final Answer: X."]), [tool], registry=registry
    ).run("Q?")
    assert [source.chunk.text for source in uncited.sources] == ["text 0", "text 1", "text 2"]

    all_sources = make_agent(
        ScriptedLLM([action("search_documents"), "Final Answer: X.\nSources: P2"]),
        [tool],
        registry=registry,
        settings=AgentSettings(verify=False, cited_sources_only=False),
    ).run("Q?")
    # Cited first, then the rest of what the agent saw.
    assert [source.chunk.text for source in all_sources.sources] == ["text 1", "text 0", "text 2"]
    assert all_sources.citations == ["P2"]


def test_agent_strips_reasoning_blocks_before_parsing():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            "<think>Let me plan the search first.</think>\n"
            "Thought: search it\n" + action("search_documents", query="x"),
            "<think>done</think>Final Answer: Answer text.",
        ]
    )

    response = make_agent(llm, [tool]).run("Question?")

    assert response.answer == "Answer text."
    assert calls == [{"query": "x"}]
    assert "<think>" not in response.steps[0].response


def test_agent_accepts_bare_json_and_fenced_tool_calls():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            '{"tool": "search_documents", "arguments": {"query": "totals"}}',
            'Action:\n```json\n{"tool": "search_documents", "arguments": {"query": "costs"}}\n```',
            "Final Answer: Done.",
        ]
    )

    response = make_agent(llm, [tool]).run("Question?")

    assert response.answer == "Done."
    assert calls == [{"query": "totals"}, {"query": "costs"}]


def test_agent_runs_several_actions_from_one_reply():
    tool, calls = make_search_tool()
    other_calls = []

    def other(**kwargs) -> ToolResult:
        other_calls.append(kwargs)
        return ToolResult(output="other result")

    other_tool = Tool(name="find_text", description="d", parameters={}, run=other)
    actions = [
        {"tool": "search_documents", "arguments": {"query": "a"}},
        {"tool": "find_text", "arguments": {"pattern": "b"}},
        {"tool": "find_text", "arguments": {"pattern": "c"}},
        {"tool": "find_text", "arguments": {"pattern": "d"}},
    ]
    llm = ScriptedLLM([f"Action: {json.dumps(actions)}", "Final Answer: Done."])

    response = make_agent(
        llm, [tool, other_tool], settings=AgentSettings(verify=False, max_parallel_actions=3)
    ).run("Question?")

    # Only the first three actions run; the fourth exceeds the parallel limit.
    assert calls == [{"query": "a"}]
    assert other_calls == [{"pattern": "b"}, {"pattern": "c"}]
    step = response.steps[0]
    assert step.tool == "search_documents"
    assert len(step.actions) == 3
    assert "Result of action 1 (search_documents):\nFound passage" in step.observation
    assert "Result of action 2 (find_text):\nother result" in step.observation


def test_agent_refuses_to_repeat_an_identical_call():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            action("search_documents", query="same"),
            action("search_documents", query="same"),
            "Final Answer: Done.",
        ]
    )

    response = make_agent(llm, [tool]).run("Question?")

    assert calls == [{"query": "same"}]
    assert response.steps[1].observation == REPEATED_CALL_NOTE.format(
        tool="search_documents", step=1
    )


def test_agent_treats_plain_text_reply_as_final_answer():
    tool, _ = make_search_tool()
    llm = ScriptedLLM(["The documents do not mention this."])

    response = make_agent(llm, [tool]).run("Question?")

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

    response = make_agent(llm, [tool]).run("Question?")

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

    response = make_agent(llm, [tool]).run("Question?")

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

    response = make_agent(llm, [tool]).run("Question?")

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

    response = make_agent(llm, [tool]).run("Question?")

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
    agent = make_agent(llm, [tool], max_steps=2)

    response = agent.run("Question?")

    assert response.answer == "Forced answer."
    assert len(calls) == 2
    # The model is warned before its last tool call, then an answer is demanded.
    assert llm.calls[1][-1]["content"].endswith(LAST_STEP_WARNING.format(remaining=1))
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

    response = make_agent(llm, [tool]).run("Question?")

    assert [source.chunk.id for source in response.sources] == ["chunk-1", "chunk-2"]


def test_agent_prefers_first_marker_when_both_are_present():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            "Thought: t\nFinal Answer: Early answer.\n" + action("search_documents", query="x"),
        ]
    )

    response = make_agent(llm, [tool]).run("Question?")

    assert response.answer.startswith("Early answer.")
    assert calls == []


def test_agent_requires_question_and_valid_configuration():
    tool, _ = make_search_tool()

    with pytest.raises(ValueError, match="question"):
        make_agent(ScriptedLLM([]), [tool]).run("   ")
    with pytest.raises(ValueError, match="max_steps"):
        make_agent(ScriptedLLM([]), [tool], max_steps=0)
    with pytest.raises(ValueError, match="tool"):
        make_agent(ScriptedLLM([]), [])


def test_agent_session_replays_previous_turns_and_records_new_ones():
    tool, _ = make_search_tool()
    session = AgentSession()
    session.record("Какая выручка в первом квартале?", "120 млн рублей.")

    llm = ScriptedLLM(["Final Answer: 138 млн рублей."])
    agent = make_agent(llm, [tool])

    response = agent.run("А во втором?", session=session)

    sent = llm.calls[0]
    assert sent[0]["role"] == "system"
    assert sent[1] == {"role": "user", "content": "Какая выручка в первом квартале?"}
    assert sent[2] == {"role": "assistant", "content": "120 млн рублей."}
    assert sent[3] == {"role": "user", "content": "А во втором?"}
    # The new exchange is recorded for the next follow-up.
    assert session.turns[-1] == ("А во втором?", "138 млн рублей.")
    assert response.answer == "138 млн рублей."


def test_agent_session_is_bounded_to_max_turns():
    session = AgentSession(max_turns=2)
    for index in range(5):
        session.record(f"q{index}", f"a{index}")

    messages = session.history_messages()

    assert len(messages) == 4  # 2 turns * (question + answer)
    assert messages[0]["content"] == "q3"
    assert messages[-1]["content"] == "a4"

    session.clear()
    assert session.history_messages() == []


def test_agent_truncates_oversized_observations():
    huge_output = "x" * (DEFAULT_MAX_OBSERVATION_CHARS + 5000)
    tool, _ = make_search_tool(output=huge_output)
    llm = ScriptedLLM(
        [
            action("search_documents", query="q"),
            "Final Answer: Done.",
        ]
    )

    response = make_agent(llm, [tool]).run("Question?")

    observation_message = llm.calls[-1][-1]["content"]
    assert len(observation_message) < DEFAULT_MAX_OBSERVATION_CHARS + 200
    assert "Observation truncated" in observation_message
    # The full output is still available to the UI via the step record.
    assert response.steps[0].observation == huge_output


def test_verification_pass_keeps_or_replaces_the_draft():
    registry = PassageRegistry()
    passage = registry.add_text("Revenue was 96 083 million.", "report.pdf")

    def run(**kwargs) -> ToolResult:
        return ToolResult(output="obs", passages=[passage])

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    verify = AgentSettings(verify=True)

    kept = FileAgent(
        llm_client=ScriptedLLM(
            [
                action("search_documents"),
                "Final Answer: Revenue was 96 083 million.\nSources: P1",
                VERIFY_KEEP_TOKEN,
            ]
        ),
        tools=[tool],
        registry=registry,
        settings=verify,
    ).run("Revenue?")
    assert kept.answer == "Revenue was 96 083 million."
    assert kept.draft_answer is None
    assert kept.steps[-1].tool == "verify_answer"

    llm = ScriptedLLM(
        [
            action("search_documents"),
            "Final Answer: Revenue was 96 million (see page 5).\nSources: P1",
            "Revenue was 96 083 million.",
        ]
    )
    edited = FileAgent(llm_client=llm, tools=[tool], registry=registry, settings=verify).run(
        "Revenue?"
    )
    assert edited.answer == "Revenue was 96 083 million."
    assert edited.draft_answer == "Revenue was 96 million (see page 5)."
    # The editor sees the question, the cited passage and the draft.
    prompt = llm.calls[-1][-1]["content"]
    assert "Revenue?" in prompt and "[P1 | file=report.pdf]" in prompt and "page 5" in prompt


def test_verification_pass_ignores_a_drastic_cut_and_survives_errors():
    registry = PassageRegistry()
    passage = registry.add_text("Long passage.", "report.pdf")

    def run(**kwargs) -> ToolResult:
        return ToolResult(output="obs", passages=[passage])

    tool = Tool(name="search_documents", description="d", parameters={}, run=run)
    long_draft = "Sentence. " * 60

    cut = FileAgent(
        llm_client=ScriptedLLM(
            [action("search_documents"), f"Final Answer: {long_draft}\nSources: P1", "OK"]
        ),
        tools=[tool],
        registry=registry,
        settings=AgentSettings(verify=True),
    ).run("Q?")
    assert cut.answer == long_draft.strip()

    class FailingVerifier(ScriptedLLM):
        def chat(self, messages):
            if messages[0]["content"].startswith("You are a meticulous editor"):
                raise RuntimeError("down")
            return super().chat(messages)

    survived = FileAgent(
        llm_client=FailingVerifier(
            [action("search_documents"), "Final Answer: Draft.\nSources: P1"]
        ),
        tools=[tool],
        registry=registry,
        settings=AgentSettings(verify=True),
    ).run("Q?")
    assert survived.answer == "Draft."


def test_system_prompt_lists_every_tool_and_document():
    tool, _ = make_search_tool()
    other = Tool(name="list_documents", description="List files.", parameters={}, run=tool.run)
    from file_agent.document import Document

    prompt = build_system_prompt(
        [tool, other], [Document(file_name="a.pdf", file_type="pdf", blocks=[])]
    )

    assert "search_documents" in prompt
    assert "list_documents" in prompt
    assert "- a.pdf (pdf)" in prompt
    assert "Final Answer" in prompt
    assert "Sources:" in prompt


def test_split_citations_and_reply_parsing():
    text, ids = _split_citations("Answer line.\nSources: P2, p5 and P2\n")
    assert (text, ids) == ("Answer line.", ["P2", "P5"])
    text, ids = _split_citations("Ответ.\n**Источники:** P1")
    assert (text, ids) == ("Ответ.", ["P1"])
    # A line about sources of something else is part of the answer.
    text, ids = _split_citations("Источники: внутренние и внешние.")
    assert (text, ids) == ("Источники: внутренние и внешние.", [])

    parsed = _parse_reply('Thought: t\nActions: [{"tool": "a", "arguments": {}}, {"tool": "b"}]')
    assert [item["tool"] for item in parsed.actions] == ["a", "b"]
    assert parsed.final is None


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
            "Final Answer: Answer from tools.\nSources: P1",
        ]
    )

    response = answer_with_agent(
        question="Question?",
        llm_client=llm,
        retriever=retriever,
        documents=[],
        settings=NO_VERIFY,
    )

    assert response.answer == "Answer from tools."
    assert [call[0] for call in retriever.search_calls] == ["totals"]
    assert [source.chunk.id for source in response.sources] == ["chunk-9"]
    assert response.citations == ["P1"]


def test_answer_with_agent_falls_back_to_single_pass_rag_when_the_loop_fails():
    class FakeRetriever:
        def index(self, chunks):
            pass

        def search(self, query: str, top_k: int = 5):
            return [make_result("chunk-1", "The report covers 2024.")]

        def clear(self):
            pass

    class BrokenThenPlain(ScriptedLLM):
        def chat(self, messages):
            if messages[0]["role"] == "system":
                raise RuntimeError("agent backend down")
            return super().chat(messages)

    llm = BrokenThenPlain(["The report covers 2024."])

    response = answer_with_agent(
        question="What does the report cover?",
        llm_client=llm,
        retriever=FakeRetriever(),
        documents=[],
        settings=NO_VERIFY,
    )

    assert response.fallback_used is True
    assert response.answer == "The report covers 2024."
    assert [source.chunk.id for source in response.sources] == ["chunk-1"]
    # The QA prompt received the retrieved context.
    assert "The report covers 2024." in llm.calls[-1][-1]["content"]

    with pytest.raises(RuntimeError, match="agent backend down"):
        answer_with_agent(
            question="Q?",
            llm_client=BrokenThenPlain([]),
            retriever=FakeRetriever(),
            documents=[],
            settings=AgentSettings(verify=False, fallback_to_rag=False),
        )


def test_agent_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_VERIFY", "off")
    monkeypatch.setenv("AGENT_CITED_SOURCES_ONLY", "false")
    monkeypatch.setenv("AGENT_MAX_PARALLEL_ACTIONS", "1")

    settings = AgentSettings.from_env()

    assert settings.max_steps == 3
    assert settings.verify is False
    assert settings.cited_sources_only is False
    assert settings.max_parallel_actions == 1
    assert settings.fingerprint()["max_steps"] == 3
    assert "prompt_version" in settings.fingerprint()


def test_budget_exhaustion_grants_one_more_call_and_strips_protocol_from_a_bare_reply():
    tool, calls = make_search_tool()
    llm = ScriptedLLM(
        [
            action("search_documents", query="first"),
            # Ignores the demand and asks for one more call: it is granted once.
            "Thought: one more\n" + action("search_documents", query="second"),
            "Thought: still thinking\n" + action("search_documents", query="third"),
        ]
    )

    response = make_agent(llm, [tool], max_steps=1).run("Question?")

    assert calls == [{"query": "first"}, {"query": "second"}]
    # The second demand repeats after the granted call.
    assert llm.calls[-1][-1]["content"].endswith(FINAL_ANSWER_DEMAND)
    # A reply that is still a tool call yields no answer text.
    assert response.answer == NO_ANSWER_MESSAGE
    assert response.steps[-1].tool is None


def test_final_demand_reply_without_marker_keeps_only_the_answer_text():
    tool, _ = make_search_tool()
    llm = ScriptedLLM(
        [
            action("search_documents", query="first"),
            "Thought: I will answer now.\nThe revenue was 100.",
        ]
    )

    response = make_agent(llm, [tool], max_steps=1).run("Question?")

    assert response.answer == "The revenue was 100."
