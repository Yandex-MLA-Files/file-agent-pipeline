"""Multi-step document agent: an LLM in a think-act-observe loop over tools.

The agent plans its own retrieval instead of answering from a single fixed
search: it decides which tool to call (fused multi-query search, exact text
search, document overview, reading sections or pages, computing over tables),
observes the result, and iterates until it can answer.

Tool calls are expressed as JSON in the model's reply and parsed here, on the
client side. This deliberately avoids server-side tool-call parsing so any
OpenAI-compatible backend works, including vLLM versions that cannot combine
tool parsing with reasoning output; ``<think>...</think>`` blocks emitted by
reasoning models are stripped before parsing.

Everything a tool shows the model is a labelled passage (``P1``, ``P2``, ...).
The Final Answer names the passages it relies on; those become the answer's
sources - what the UI lists and what an evaluation judges the answer against -
so the answer text itself stays free of file names and page numbers.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from file_agent.agent.passages import Passage, PassageRegistry
from file_agent.agent.tools import Tool, ToolError
from file_agent.document import Document
from file_agent.llm.base import ChatLLMClient
from file_agent.retrieval import Retriever, SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8
# Previous question/answer pairs kept in the conversation for follow-ups.
# Only final answers are replayed, never tool traffic: session memory should
# carry the dialogue, not stale observations that would crowd the context.
DEFAULT_MAX_SESSION_TURNS = 4
# Upper bound for a single observation fed back to the model. Protects the
# context window from oversized tool output (e.g. several large parent
# passages at once).
DEFAULT_MAX_OBSERVATION_CHARS = 14000
# Independent tool calls the model may issue in one reply (a JSON list).
DEFAULT_MAX_PARALLEL_ACTIONS = 3
# Contexts exported when the answer cites nothing (or citations are disabled).
MAX_UNCITED_SOURCES = 8
OBSERVATION_TRUNCATION_NOTE = (
    "\n[Observation truncated. Narrow the query, lower top_k or read a specific part.]"
)

NO_ANSWER_MESSAGE = "The agent could not produce an answer from the documents."
# Bump when the system prompt or the answer protocol changes; recorded with
# generated datasets so cached rows from another prompt are not reused.
PROMPT_VERSION = "agent-v2"

FORMAT_REMINDER = (
    "Your reply did not match the required format. Reply either with\n"
    'Action: {"tool": "<tool name>", "arguments": {...}}\n'
    "or with\n"
    "Final Answer: <answer>\nSources: <passage ids>"
)

FINAL_ANSWER_DEMAND = (
    "You have used all available tool calls. Do not call any more tools. "
    "Give the Final Answer now, using only the passages above, and end with the "
    "Sources line. If they are insufficient, say the documents do not contain "
    "enough information."
)

LAST_STEP_WARNING = "\n\n[You have {remaining} tool call(s) left before you must answer.]"

REPEATED_CALL_NOTE = (
    "You already ran {tool} with exactly these arguments in step {step}; the result "
    "would be identical. Use different arguments, another tool, or give the Final Answer."
)

VERIFY_SYSTEM_PROMPT = (
    "You are a meticulous editor. You check a draft answer against the passages "
    "it was written from and return the corrected answer."
)

VERIFY_KEEP_TOKEN = "KEEP"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THOUGHT_LINE = re.compile(r"Thought:\s*(.+)")
_ACTION_MARKER = re.compile(r"Actions?\s*:", re.IGNORECASE)
_FINAL_MARKER = re.compile(r"Final Answer\s*:", re.IGNORECASE)
_SOURCES_LINE = re.compile(
    r"^\s*\**\s*(?:Sources?|Источники?)\s*\**\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE
)
_PASSAGE_ID = re.compile(r"\bP\s*(\d+)\b", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, low: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(low, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


@dataclass(frozen=True)
class AgentSettings:
    """Knobs of the loop, all overridable from the environment."""

    max_steps: int = DEFAULT_MAX_STEPS
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    max_parallel_actions: int = DEFAULT_MAX_PARALLEL_ACTIONS
    # Run the editor pass over the draft answer (one extra model call).
    verify: bool = True
    # Export only the passages the answer cites as its sources; ``False``
    # exports everything the agent saw (bounded).
    cited_sources_only: bool = True
    # Fall back to single-pass RAG over the collected passages when the loop
    # crashes or ends without an answer.
    fallback_to_rag: bool = True
    trace_dir: str | None = None

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            max_steps=_env_int("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS),
            max_observation_chars=_env_int(
                "AGENT_MAX_OBSERVATION_CHARS", DEFAULT_MAX_OBSERVATION_CHARS, low=1000
            ),
            max_parallel_actions=_env_int(
                "AGENT_MAX_PARALLEL_ACTIONS", DEFAULT_MAX_PARALLEL_ACTIONS
            ),
            verify=_env_flag("AGENT_VERIFY", True),
            cited_sources_only=_env_flag("AGENT_CITED_SOURCES_ONLY", True),
            fallback_to_rag=_env_flag("AGENT_FALLBACK_TO_RAG", True),
            trace_dir=(os.getenv("AGENT_TRACE_DIR") or "").strip() or None,
        )

    def fingerprint(self) -> dict[str, Any]:
        """What changes the answers; recorded with generated datasets."""
        return {
            "prompt_version": PROMPT_VERSION,
            "max_steps": self.max_steps,
            "max_observation_chars": self.max_observation_chars,
            "max_parallel_actions": self.max_parallel_actions,
            "verify": self.verify,
            "cited_sources_only": self.cited_sources_only,
        }


def build_system_prompt(
    tools: list[Tool],
    documents: list[Document] | None = None,
    max_parallel_actions: int = DEFAULT_MAX_PARALLEL_ACTIONS,
) -> str:
    tool_lines = "\n".join(tool.describe() for tool in tools)
    if documents:
        document_lines = "\n".join(
            f"- {document.file_name} ({document.file_type})" for document in documents
        )
        documents_block = f"Uploaded documents:\n{document_lines}\n\n"
    else:
        documents_block = ""
    parallel = (
        f"You may issue up to {max_parallel_actions} independent calls at once as a JSON list: "
        'Action: [{"tool": "...", "arguments": {...}}, {"tool": "...", "arguments": {...}}]\n'
        if max_parallel_actions > 1
        else ""
    )
    return (
        "You are a document analysis agent. You answer the user's question using only "
        "the content of the uploaded documents, which you access through tools. You "
        "never answer from memory or general knowledge.\n\n"
        f"{documents_block}"
        f"Tools:\n{tool_lines}\n\n"
        "How to work:\n"
        "1. Start with search_documents: put the key terms of the question in 'query' "
        "and 2-3 alternative formulations in 'queries' (synonyms, the wording the "
        "document itself would use, the English term for a Russian question and vice "
        "versa).\n"
        "2. When the question names exact things - numbers, codes, identifiers, names, "
        "dates, rare terms, quoted phrases - also use find_text with that exact text; it "
        "is precise where semantic search is fuzzy.\n"
        "3. When the question involves two documents, query each document separately "
        "with file_name, then combine what you found.\n"
        "4. For questions about a document's structure, sections, slides, introduction "
        "or conclusion, or 'how many ...', call list_documents and then read_section, "
        "read_pages or read_document.\n"
        "5. For spreadsheets and tables (totals, counts, averages, maxima, unique values, "
        "matches across files, exact row lookups) use query_table with pandas code; call "
        "it with empty code first to see the columns.\n"
        "6. Do arithmetic with calculate, not in your head.\n"
        "7. If an attempt finds nothing, try once or twice more with different words or "
        "another tool. If the documents genuinely do not contain the information, say so "
        "in the Final Answer and briefly mention what related information they do "
        "contain.\n"
        "8. Never repeat a call you already made with the same arguments.\n\n"
        "Every passage you receive is labelled [P<n> | file=... | ...]. Remember the ids "
        "of the passages you use.\n\n"
        "Each of your replies must use exactly one of the two formats.\n\n"
        "To call a tool:\n"
        "Thought: <one sentence - what you need and why>\n"
        'Action: {"tool": "<tool name>", "arguments": {"<name>": <value>}}\n'
        f"{parallel}"
        "\n"
        "To answer the user:\n"
        "Thought: <one sentence>\n"
        "Final Answer: <the answer>\n"
        "Sources: <comma-separated ids of every passage the answer relies on, e.g. P2, P5>\n\n"
        "Final Answer rules:\n"
        "- Write in the language of the question (a Russian question gets a Russian "
        "answer).\n"
        "- Answer completely: every part of the question, every item of a list, the "
        "exact numbers, names and wording as they appear in the passages. Prefer the "
        "document's own terms.\n"
        "- Use only facts from the passages you have seen. Never add outside knowledge, "
        "assumptions or estimates; quote numbers exactly.\n"
        "- Do not put file names, page or slide numbers, section numbers, tool names or "
        "passage ids inside the answer text; the Sources line is the only place for "
        "them. Do not describe how you searched.\n"
        "- Be direct: no preamble, no restating the question. Use a short list when the "
        "answer has several items.\n"
        "- The Sources line is mandatory and must be the last line.\n"
        "- Earlier questions and answers may precede the current question; use them to "
        "resolve references (like 'and in the second quarter?'), but ground every new "
        "fact in passages from this run."
    )


@dataclass
class AgentStep:
    """One think-act-observe iteration, kept for tracing and the UI."""

    response: str
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None
    # Set when the reply carried several actions; ``tool``/``arguments`` then
    # describe the first one.
    actions: list[dict[str, Any]] | None = None
    elapsed_seconds: float | None = None


@dataclass
class AgentResponse:
    answer: str
    sources: list[SearchResult]
    steps: list[AgentStep]
    # Ids of the passages the answer cited, in citation order (empty when the
    # model cited nothing and every collected passage was exported).
    citations: list[str] = field(default_factory=list)
    draft_answer: str | None = None
    fallback_used: bool = False
    passages_seen: int = 0


@dataclass
class AgentSession:
    """Bounded conversation memory that enables follow-up questions.

    Keeps the last ``max_turns`` question/answer pairs and replays them as
    plain chat turns before the current question. Tool calls and observations
    from previous runs are deliberately not replayed: they are stale working
    state, and replaying them would crowd the context window without adding
    grounding (the agent re-queries the documents instead).
    """

    max_turns: int = DEFAULT_MAX_SESSION_TURNS
    turns: list[tuple[str, str]] = field(default_factory=list)

    def history_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for question, answer in self.turns[-self.max_turns :]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def record(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))

    def clear(self) -> None:
        self.turns.clear()


@dataclass
class _ParsedReply:
    thought: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    final: str | None = None


class FileAgent:
    """Runs the think-act-observe loop until an answer or the step budget."""

    def __init__(
        self,
        llm_client: ChatLLMClient,
        tools: list[Tool],
        max_steps: int | None = None,
        registry: PassageRegistry | None = None,
        documents: list[Document] | None = None,
        settings: AgentSettings | None = None,
    ) -> None:
        self._settings = settings or AgentSettings()
        if max_steps is not None:
            self._settings = replace(self._settings, max_steps=max_steps)
        if self._settings.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not tools:
            raise ValueError("at least one tool is required")

        self._llm = llm_client
        self._tools = {tool.name: tool for tool in tools}
        self._registry = registry if registry is not None else PassageRegistry()
        self._system_prompt = build_system_prompt(
            tools, documents, max_parallel_actions=self._settings.max_parallel_actions
        )

    @property
    def registry(self) -> PassageRegistry:
        return self._registry

    def run(self, question: str, session: AgentSession | None = None) -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("question is required")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            *(session.history_messages() if session else []),
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        collected: list[Passage] = []
        collected_keys: set[str] = set()
        executed: dict[str, int] = {}
        max_steps = self._settings.max_steps

        with tracer.start_as_current_span("file_agent.agent_run") as span:
            span.set_attribute("file_agent.question", question)
            span.set_attribute("file_agent.max_steps", max_steps)
            span.set_attribute("file_agent.session_turns", len(session.turns) if session else 0)

            for step_index in range(1, max_steps + 1):
                started = time.perf_counter()
                reply = _visible_text(self._llm.chat(messages))
                parsed = _parse_reply(reply)

                if parsed.final is not None:
                    step = AgentStep(response=reply, thought=parsed.thought)
                    step.elapsed_seconds = time.perf_counter() - started
                    steps.append(step)
                    return self._finish(span, parsed.final, collected, steps, session, question)

                step = AgentStep(response=reply, thought=parsed.thought)
                if not parsed.actions:
                    step.observation = FORMAT_REMINDER
                    logger.info("Agent reply did not contain an action or a final answer")
                else:
                    actions = parsed.actions[: self._settings.max_parallel_actions]
                    step.tool = str(actions[0].get("tool", ""))
                    step.arguments = _arguments_of(actions[0])
                    if len(actions) > 1:
                        step.actions = actions
                    observations: list[str] = []
                    for position, action in enumerate(actions, start=1):
                        tool_name = str(action.get("tool", ""))
                        arguments = _arguments_of(action)
                        signature = json.dumps(
                            {"tool": tool_name, "arguments": arguments}, sort_keys=True, default=str
                        )
                        if signature in executed:
                            observation = REPEATED_CALL_NOTE.format(
                                tool=tool_name, step=executed[signature]
                            )
                        else:
                            executed[signature] = step_index
                            observation, result_passages = self._execute(tool_name, arguments)
                            for passage in result_passages:
                                if passage.id not in collected_keys:
                                    collected_keys.add(passage.id)
                                    collected.append(passage)
                        if len(actions) > 1:
                            observation = (
                                f"Result of action {position} ({tool_name}):\n{observation}"
                            )
                        observations.append(observation)
                    step.observation = "\n\n".join(observations)

                step.elapsed_seconds = time.perf_counter() - started
                steps.append(step)
                observation = _bounded_observation(
                    step.observation or "", self._settings.max_observation_chars
                )
                remaining = max_steps - step_index
                if 0 < remaining <= 1:
                    observation += LAST_STEP_WARNING.format(remaining=remaining)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            # Step budget exhausted: demand an answer from what was observed.
            messages.append({"role": "user", "content": FINAL_ANSWER_DEMAND})
            started = time.perf_counter()
            reply = _visible_text(self._llm.chat(messages))
            parsed = _parse_reply(reply)
            steps.append(
                AgentStep(
                    response=reply,
                    thought=parsed.thought,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            answer = parsed.final if parsed.final is not None else reply
            return self._finish(span, answer or "", collected, steps, session, question)

    def _finish(
        self,
        span: Any,
        raw_answer: str,
        collected: list[Passage],
        steps: list[AgentStep],
        session: AgentSession | None,
        question: str,
    ) -> AgentResponse:
        answer, citations = _split_citations(raw_answer)
        cited = [self._registry.get(passage_id) for passage_id in citations]
        cited_passages = [passage for passage in cited if passage is not None]
        resolved_ids = [passage.id for passage in cited_passages]

        if cited_passages and self._settings.cited_sources_only:
            source_passages = cited_passages
        else:
            source_passages = cited_passages + [
                passage for passage in collected if passage.id not in resolved_ids
            ]
            source_passages = source_passages[: max(MAX_UNCITED_SOURCES, len(cited_passages))]

        draft = answer.strip()
        final_answer = draft
        if draft and self._settings.verify and source_passages:
            final_answer = self._verify(question, draft, source_passages, steps)

        final_answer = final_answer.strip() or NO_ANSWER_MESSAGE
        if session is not None:
            session.record(question, final_answer)
        span.set_attribute("file_agent.step_count", len(steps))
        span.set_attribute("file_agent.source_count", len(source_passages))
        span.set_attribute("file_agent.citation_count", len(resolved_ids))
        span.set_attribute("file_agent.answer_length", len(final_answer))
        logger.info(
            "Agent finished in %d step(s) with %d cited / %d seen passage(s)",
            len(steps),
            len(resolved_ids),
            len(self._registry),
        )
        return AgentResponse(
            answer=final_answer,
            sources=[passage.as_search_result() for passage in source_passages],
            steps=steps,
            citations=resolved_ids,
            draft_answer=draft if final_answer != draft else None,
            passages_seen=len(self._registry),
        )

    def _verify(
        self,
        question: str,
        draft: str,
        passages: list[Passage],
        steps: list[AgentStep],
    ) -> str:
        """One editor pass: drop unsupported claims, add missed specifics."""
        rendered = "\n\n".join(passage.render(max_chars=3500) for passage in passages)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Passages the draft was written from:\n{rendered}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Check the draft:\n"
            "1. Every statement must be supported by the passages. Remove or correct "
            "anything the passages do not say; do not add outside knowledge.\n"
            "2. It must answer every part of the question with all the specifics the "
            "passages provide (exact numbers, names, list items, wording); add what is "
            "missing from the passages.\n"
            "3. It must be in the language of the question, direct, without file names, "
            "page or section numbers, passage ids or remarks about the search.\n"
            "If the draft already satisfies all three, reply with exactly "
            f"{VERIFY_KEEP_TOKEN}. Otherwise reply with the corrected answer text only - "
            "no explanations, no headings, no Sources line."
        )
        started = time.perf_counter()
        try:
            reply = _visible_text(
                self._llm.chat(
                    [
                        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
            )
        except Exception:  # noqa: BLE001 - verification is best effort
            logger.warning("Answer verification failed; keeping the draft", exc_info=True)
            return draft
        steps.append(
            AgentStep(
                response=reply,
                thought="verification",
                tool="verify_answer",
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        cleaned, _ = _split_citations(reply)
        cleaned = _strip_final_marker(cleaned).strip()
        if not cleaned or cleaned.upper().strip("*. ") == VERIFY_KEEP_TOKEN:
            return draft
        if len(cleaned) < 0.3 * len(draft) and len(draft) > 200:
            # A drastic cut is more often a misfire than an edit.
            logger.info("Verifier reply much shorter than the draft; keeping the draft")
            return draft
        return cleaned

    def _execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, list[Passage]]:
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
            except TypeError as exc:
                # Wrong keyword set for a tool without **extra handling.
                span.set_attribute("file_agent.tool_error", str(exc))
                return f"Tool '{tool_name}' rejected the arguments: {exc}", []
            except Exception as exc:  # defensive: a tool bug must not kill the run
                logger.warning("Tool %s failed", tool_name, exc_info=True)
                span.set_attribute("file_agent.tool_error", str(exc))
                return f"Tool '{tool_name}' failed: {exc}", []

            span.set_attribute("file_agent.observation_length", len(result.output))
            passages = list(result.passages) or [
                self._registry.add_result(source, tool=tool_name) for source in result.sources
            ]
            return result.output, passages


def answer_with_agent(
    question: str,
    llm_client: ChatLLMClient,
    retriever: Retriever,
    documents: list[Document],
    max_steps: int | None = None,
    tools: list[Tool] | None = None,
    session: AgentSession | None = None,
    settings: AgentSettings | None = None,
) -> AgentResponse:
    """Answer a question about already-indexed documents with the agent loop.

    The loop is wrapped in a safety net: when it raises or ends without an
    answer, the collected passages (or a plain retrieval) feed the single-pass
    QA prompt so the caller always gets an answer.
    """
    from file_agent.agent.tools import build_default_tools

    active_settings = settings or AgentSettings.from_env()
    registry = PassageRegistry()
    agent = FileAgent(
        llm_client=llm_client,
        tools=tools if tools is not None else build_default_tools(retriever, documents, registry),
        max_steps=max_steps,
        registry=registry,
        documents=documents,
        settings=active_settings,
    )
    started = time.perf_counter()
    try:
        response = agent.run(question, session=session)
    except Exception as exc:  # noqa: BLE001 - the fallback below is the whole point
        if not active_settings.fallback_to_rag:
            raise
        logger.warning(
            "Agent loop failed (%s); falling back to single-pass RAG", exc, exc_info=True
        )
        response = _rag_fallback(
            question, llm_client, retriever, registry, steps=[], reason=str(exc)
        )
    else:
        if response.answer == NO_ANSWER_MESSAGE and active_settings.fallback_to_rag:
            response = _rag_fallback(
                question, llm_client, retriever, registry, response.steps, reason="no answer"
            )
    _write_trace(active_settings.trace_dir, question, response, time.perf_counter() - started)
    return response


def _rag_fallback(
    question: str,
    llm_client: ChatLLMClient,
    retriever: Retriever,
    registry: PassageRegistry,
    steps: list[AgentStep],
    reason: str,
) -> AgentResponse:
    from file_agent.qa import answer_question_with_context

    sources = [passage.as_search_result() for passage in registry.all()][:MAX_UNCITED_SOURCES]
    if not sources:
        try:
            sources = retriever.search(query=question, top_k=5)
        except Exception:  # noqa: BLE001
            logger.warning("Fallback retrieval failed", exc_info=True)
            sources = []
    answer = answer_question_with_context(question=question, results=sources, llm_client=llm_client)
    steps = [*steps, AgentStep(response=answer, thought=f"fallback to single-pass RAG ({reason})")]
    return AgentResponse(
        answer=answer,
        sources=sources,
        steps=steps,
        citations=[],
        fallback_used=True,
        passages_seen=len(registry),
    )


def _write_trace(
    trace_dir: str | None, question: str, response: AgentResponse, elapsed: float
) -> None:
    if not trace_dir:
        return
    try:
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "question": question,
            "answer": response.answer,
            "draft_answer": response.draft_answer,
            "citations": response.citations,
            "fallback_used": response.fallback_used,
            "passages_seen": response.passages_seen,
            "elapsed_seconds": round(elapsed, 2),
            "sources": [
                {
                    "id": source.chunk.id,
                    "file": source.chunk.metadata.get("source_file"),
                    "tool": source.chunk.metadata.get("tool"),
                }
                for source in response.sources
            ],
            "steps": [
                {
                    "thought": step.thought,
                    "tool": step.tool,
                    "arguments": step.arguments,
                    "actions": step.actions,
                    "observation": step.observation,
                    "response": step.response,
                    "elapsed_seconds": round(step.elapsed_seconds or 0.0, 2),
                }
                for step in response.steps
            ],
        }
        with (path / "agent_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - tracing must never break answering
        logger.warning("Could not write the agent trace", exc_info=True)


def _bounded_observation(observation: str, limit: int) -> str:
    if len(observation) <= limit:
        return observation
    return observation[:limit] + OBSERVATION_TRUNCATION_NOTE


def _visible_text(reply: str) -> str:
    """Drop reasoning blocks so parsing sees only the model's visible output."""
    text = _THINK_BLOCK.sub("", reply or "")
    # An opening tag without a closing one means the reasoning was cut off;
    # nothing after it is a deliberate reply.
    unclosed = text.find("<think>")
    if unclosed != -1:
        text = text[:unclosed]
    return text.strip()


def _split_citations(answer: str) -> tuple[str, list[str]]:
    """Separate the ``Sources: P1, P4`` line(s) from the answer text."""
    ordered: list[str] = []
    text = answer
    # Only a line that actually lists ids is a citation line: an answer may
    # legitimately talk about "sources" of something else.
    for match in reversed(list(_SOURCES_LINE.finditer(answer))):
        if _PASSAGE_ID.search(match.group(1)):
            text = text[: match.start()] + text[match.end() :]
    for match in _SOURCES_LINE.finditer(answer):
        for number in _PASSAGE_ID.findall(match.group(1)):
            passage_id = f"P{number}"
            if passage_id not in ordered:
                ordered.append(passage_id)
    return text.strip(), ordered


def _strip_final_marker(text: str) -> str:
    match = _FINAL_MARKER.search(text)
    if match and match.start() < 40:
        return text[match.end() :]
    return text


def _arguments_of(action: dict[str, Any]) -> dict[str, Any]:
    arguments = action.get("arguments")
    if arguments is None:
        arguments = action.get("args") or action.get("parameters") or action.get("input")
    return arguments if isinstance(arguments, dict) else {}


def _parse_reply(reply: str) -> _ParsedReply:
    parsed = _ParsedReply()

    thought_match = _THOUGHT_LINE.search(reply)
    if thought_match:
        parsed.thought = thought_match.group(1).strip()

    action_match = _ACTION_MARKER.search(reply)
    final_match = _FINAL_MARKER.search(reply)

    # When both markers are present, honor whichever the model wrote first.
    if action_match and (not final_match or action_match.start() < final_match.start()):
        actions = _extract_actions(reply, action_match.end())
        if actions:
            parsed.actions = actions
            return parsed
        if final_match:
            parsed.final = reply[final_match.end() :].strip()
        return parsed

    if final_match:
        parsed.final = reply[final_match.end() :].strip()
        return parsed

    # No markers: some models emit a bare JSON tool call or a plain answer.
    actions = _extract_actions(reply, 0)
    if actions:
        parsed.actions = actions
    elif reply:
        parsed.final = reply
    return parsed


def _extract_actions(text: str, start: int) -> list[dict[str, Any]]:
    """Find the first JSON object (or list of objects) with a 'tool' key at or after start."""
    region = text[start:]
    fenced = _CODE_FENCE.search(region)
    if fenced and fenced.start() < 20:
        region = fenced.group(1)
    position = 0
    while True:
        opening_obj = region.find("{", position)
        opening_list = region.find("[", position)
        candidates = [index for index in (opening_obj, opening_list) if index != -1]
        if not candidates:
            return []
        opening = min(candidates)
        candidate = _balanced_json(region, opening)
        if candidate is not None:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict) and isinstance(data.get("tool"), str):
                return [data]
            if isinstance(data, list):
                actions = [
                    item
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("tool"), str)
                ]
                if actions:
                    return actions
        position = opening + 1


def _balanced_json(text: str, opening: int) -> str | None:
    """Return the substring of the balanced {...} or [...] value starting at opening."""
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
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1]

    return None
