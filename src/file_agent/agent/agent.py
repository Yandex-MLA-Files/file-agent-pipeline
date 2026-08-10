"""Multi-step document agent: an LLM in a think-act-observe loop over tools.

The agent plans its own retrieval instead of answering from a single fixed
search: it decides which tool to call (search, document overview, reading a
whole section), observes the result, and iterates until it can answer.

Tool calls are expressed as JSON in the model's reply and parsed here, on the
client side. This deliberately avoids server-side tool-call parsing so any
OpenAI-compatible backend works, including vLLM versions that cannot combine
tool parsing with reasoning output; ``<think>...</think>`` blocks emitted by
reasoning models are stripped before parsing.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from file_agent.agent.tools import Tool, ToolError
from file_agent.document import Document
from file_agent.llm.base import ChatLLMClient
from file_agent.retrieval import Retriever, SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6

NO_ANSWER_MESSAGE = "The agent could not produce an answer from the documents."

FORMAT_REMINDER = (
    "Your reply did not match the required format. Reply either with\n"
    'Action: {"tool": "<tool name>", "arguments": {...}}\n'
    "or with\n"
    "Final Answer: <answer>"
)

FINAL_ANSWER_DEMAND = (
    "You have used all available tool calls. Do not call any more tools. "
    "Give the Final Answer now, using only the observations above. If they "
    "are insufficient, say the documents do not contain enough information."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THOUGHT_LINE = re.compile(r"Thought:\s*(.+)")
_ACTION_MARKER = re.compile(r"Action\s*:", re.IGNORECASE)
_FINAL_MARKER = re.compile(r"Final Answer\s*:", re.IGNORECASE)


def build_system_prompt(tools: list[Tool]) -> str:
    tool_lines = "\n".join(tool.describe() for tool in tools)
    return (
        "You are a document analysis agent. You answer the user's question "
        "using only the content of the uploaded documents, which you access "
        "through tools.\n\n"
        f"Tools:\n{tool_lines}\n\n"
        "You work in steps. Each of your replies must use exactly one of the "
        "two formats.\n\n"
        "To call a tool:\n"
        "Thought: <one sentence - what you need and why>\n"
        'Action: {"tool": "<tool name>", "arguments": {"<name>": <value>}}\n\n'
        "To answer the user:\n"
        "Thought: <one sentence>\n"
        "Final Answer: <the answer>\n\n"
        "Rules:\n"
        "- Call one tool per reply, then wait for its Observation.\n"
        "- Use only facts from Observations; never invent document content.\n"
        "- If the observations do not contain the answer, reformulate the "
        "search once or twice; if that fails, state in the Final Answer that "
        "the documents do not contain enough information.\n"
        "- Write the Final Answer in the same language as the user's "
        "question, and name the source files (and sections or pages when "
        "known) the answer is based on.\n"
        "- Keep the Final Answer short and factual."
    )


@dataclass
class AgentStep:
    """One think-act-observe iteration, kept for tracing and the UI."""

    response: str
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResponse:
    answer: str
    sources: list[SearchResult]
    steps: list[AgentStep]


@dataclass
class _ParsedReply:
    thought: str | None = None
    action: dict[str, Any] | None = None
    final: str | None = None


class FileAgent:
    """Runs the think-act-observe loop until an answer or the step budget."""

    def __init__(
        self,
        llm_client: ChatLLMClient,
        tools: list[Tool],
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not tools:
            raise ValueError("at least one tool is required")

        self._llm = llm_client
        self._tools = {tool.name: tool for tool in tools}
        self._max_steps = max_steps
        self._system_prompt = build_system_prompt(tools)

    def run(self, question: str) -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("question is required")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        sources: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()

        with tracer.start_as_current_span("file_agent.agent_run") as span:
            span.set_attribute("file_agent.question", question)
            span.set_attribute("file_agent.max_steps", self._max_steps)

            for _ in range(self._max_steps):
                reply = _visible_text(self._llm.chat(messages))
                parsed = _parse_reply(reply)

                if parsed.final is not None:
                    steps.append(AgentStep(response=reply, thought=parsed.thought))
                    return self._finish(span, parsed.final, sources, steps)

                step = AgentStep(response=reply, thought=parsed.thought)
                if parsed.action is None:
                    step.observation = FORMAT_REMINDER
                    logger.info("Agent reply did not contain an action or a final answer")
                else:
                    step.tool = str(parsed.action.get("tool", ""))
                    arguments = parsed.action.get("arguments") or {}
                    step.arguments = arguments if isinstance(arguments, dict) else {}
                    step.observation, step_sources = self._execute(step.tool, step.arguments)
                    for result in step_sources:
                        if result.chunk.id not in seen_chunk_ids:
                            seen_chunk_ids.add(result.chunk.id)
                            sources.append(result)

                steps.append(step)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation: {step.observation}"})

            # Step budget exhausted: demand an answer from what was observed.
            messages.append({"role": "user", "content": FINAL_ANSWER_DEMAND})
            reply = _visible_text(self._llm.chat(messages))
            parsed = _parse_reply(reply)
            steps.append(AgentStep(response=reply, thought=parsed.thought))
            answer = parsed.final if parsed.final is not None else reply
            return self._finish(span, answer or NO_ANSWER_MESSAGE, sources, steps)

    def _finish(
        self,
        span: Any,
        answer: str,
        sources: list[SearchResult],
        steps: list[AgentStep],
    ) -> AgentResponse:
        answer = answer.strip() or NO_ANSWER_MESSAGE
        span.set_attribute("file_agent.step_count", len(steps))
        span.set_attribute("file_agent.source_count", len(sources))
        span.set_attribute("file_agent.answer_length", len(answer))
        logger.info("Agent finished in %d step(s) with %d source(s)", len(steps), len(sources))
        return AgentResponse(answer=answer, sources=sources, steps=steps)

    def _execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, list[SearchResult]]:
        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return f"Unknown tool '{tool_name}'. Available tools: {available}.", []

        with tracer.start_as_current_span("file_agent.agent_tool") as span:
            span.set_attribute("file_agent.tool", tool_name)
            span.set_attribute("file_agent.tool_arguments", json.dumps(arguments, default=str))
            try:
                result = tool.run(**arguments)
            except ToolError as exc:
                span.set_attribute("file_agent.tool_error", str(exc))
                return str(exc), []
            except Exception as exc:  # defensive: a tool bug must not kill the run
                logger.warning("Tool %s failed", tool_name, exc_info=True)
                span.set_attribute("file_agent.tool_error", str(exc))
                return f"Tool '{tool_name}' failed: {exc}", []

            span.set_attribute("file_agent.observation_length", len(result.output))
            return result.output, result.sources


def answer_with_agent(
    question: str,
    llm_client: ChatLLMClient,
    retriever: Retriever,
    documents: list[Document],
    max_steps: int = DEFAULT_MAX_STEPS,
    tools: list[Tool] | None = None,
) -> AgentResponse:
    """Answer a question about already-indexed documents with the agent loop."""
    from file_agent.agent.tools import build_default_tools

    agent = FileAgent(
        llm_client=llm_client,
        tools=tools if tools is not None else build_default_tools(retriever, documents),
        max_steps=max_steps,
    )
    return agent.run(question)


def _visible_text(reply: str) -> str:
    """Drop reasoning blocks so parsing sees only the model's visible output."""
    text = _THINK_BLOCK.sub("", reply)
    # An opening tag without a closing one means the reasoning was cut off;
    # nothing after it is a deliberate reply.
    unclosed = text.find("<think>")
    if unclosed != -1:
        text = text[:unclosed]
    return text.strip()


def _parse_reply(reply: str) -> _ParsedReply:
    parsed = _ParsedReply()

    thought_match = _THOUGHT_LINE.search(reply)
    if thought_match:
        parsed.thought = thought_match.group(1).strip()

    action_match = _ACTION_MARKER.search(reply)
    final_match = _FINAL_MARKER.search(reply)

    # When both markers are present, honor whichever the model wrote first.
    if action_match and (not final_match or action_match.start() < final_match.start()):
        action = _extract_action(reply, action_match.end())
        if action is not None:
            parsed.action = action
            return parsed
        if final_match:
            parsed.final = reply[final_match.end() :].strip()
        return parsed

    if final_match:
        parsed.final = reply[final_match.end() :].strip()
        return parsed

    # No markers: some models emit a bare JSON tool call or a plain answer.
    action = _extract_action(reply, 0)
    if action is not None:
        parsed.action = action
    elif reply:
        parsed.final = reply
    return parsed


def _extract_action(text: str, start: int) -> dict[str, Any] | None:
    """Find the first balanced JSON object with a 'tool' key at or after start."""
    position = start
    while True:
        opening = text.find("{", position)
        if opening == -1:
            return None

        candidate = _balanced_json(text, opening)
        if candidate is not None:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict) and isinstance(data.get("tool"), str):
                return data
        position = opening + 1


def _balanced_json(text: str, opening: int) -> str | None:
    """Return the substring of the balanced {...} object starting at opening."""
    depth = 0
    in_string = False
    escaped = False

    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1]

    return None
