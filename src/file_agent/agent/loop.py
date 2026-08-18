import json
import logging
from dataclasses import dataclass
from typing import Any

from file_agent.agent.observability import finish_trace, log_generation, log_tool_call
from file_agent.agent.tools import Tool, ToolResult
from file_agent.llm.base import LLMClient, ToolCall, ToolCallResponse
from file_agent.retrieval import SearchResult
from file_agent.telemetry import tracer

logger = logging.getLogger(__name__)

MAX_ITERATIONS_DEFAULT = 10
MAX_VERIFICATION_EVIDENCE_CHARS = 6000
CONTEXT_HEADROOM_FRACTION = 0.75

SYSTEM_PROMPT = (
    "You are a document question-answering agent. Use the available tools to "
    "find and compute the answer. Once you have enough information, respond "
    "directly with a concise final answer in plain text (no further tool "
    "calls), entirely in the same language as the question - do not switch "
    "languages mid-answer or mix in words from another language, even for "
    "technical terms. If the tools genuinely don't provide enough "
    "information, say so instead of guessing. For a yes/no or 'does the "
    "material cover X' question, a couple of differently-phrased searches "
    "that all come back irrelevant is itself the answer - it likely means "
    "the material doesn't cover X, so say that with confidence instead of "
    "trying yet another rephrasing of the same search. Conversation history "
    "is provided only to resolve references in the current question, like "
    "'and the second one?' or 'what about the other document?' - it is not "
    "evidence. Never restate a document fact from an earlier answer in this "
    "conversation without checking the documents again for the current "
    "question; always use at least one tool before making a new factual "
    "claim, even if it looks like the previous answer already covers it. For "
    "a 'how many' or exact-count question (how many references are cited, "
    "how many times a term appears, how many entries a list has), do not "
    "count by eye from search results or a page of text - that is unreliable "
    "- use run_python to count precisely over the document's full extracted "
    "text or its tables."
)


def _fits_context(llm_client: LLMClient, messages: list[dict[str, Any]]) -> bool:

    context_length = getattr(llm_client, "context_length", None)
    if context_length is None:
        return True
    estimated_tokens = len(json.dumps(messages, ensure_ascii=False)) // 2
    return estimated_tokens <= context_length * CONTEXT_HEADROOM_FRACTION


def _build_system_prompt(history_summary: str | None) -> str:
    if not history_summary:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\nSummary of earlier turns in this conversation, no "
        f"longer kept verbatim: {history_summary}"
    )


def _verify_answer(
    llm_client: LLMClient,
    question: str,
    answer: str,
    sources: list[SearchResult],
    tool_schemas: list[dict[str, Any]],
) -> str:

    if not sources:
        return answer

    seen_ids: set[str] = set()
    pieces: list[str] = []
    for result in sources:
        if result.chunk.id in seen_ids:
            continue
        seen_ids.add(result.chunk.id)
        pieces.append(result.chunk.text)
    evidence = "\n\n".join(pieces)[:MAX_VERIFICATION_EVIDENCE_CHARS]

    prompt = (
        "You drafted an answer to a question using the evidence below. Check "
        "that every claim in the draft is actually supported by that "
        "evidence. If it is fully supported, repeat the draft answer "
        "unchanged. If any part is not supported, rewrite the answer so it "
        "states only what the evidence actually supports - say so "
        "explicitly if that means the question can't be fully answered. "
        "Reply with only the final answer text, in the same language as the "
        "question - no preamble, no explanation of what changed.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence}\n\nDraft answer: {answer}"
    )
    verification_messages = [{"role": "user", "content": prompt}]
    try:
        response = llm_client.generate_with_tools(
            verification_messages, tool_schemas, tool_choice="none"
        )
    except Exception:
        logger.warning("Answer verification call failed; keeping the draft answer", exc_info=True)
        return answer

    revised = (response.content or "").strip()
    log_generation(
        model=getattr(llm_client, "model", "unknown"),
        input_messages=verification_messages,
        output=revised,
        reasoning=response.reasoning,
        usage=response.usage,
    )
    return revised or answer


@dataclass
class AgentResponse:
    answer: str
    sources: list[SearchResult]
    iterations: int
    last_usage: dict[str, int] | None = None


def run_react_agent(
    question: str,
    llm_client: LLMClient,
    tools: list[Tool],
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    conversation_history: list[dict[str, Any]] | None = None,
    history_summary: str | None = None,
    verify_answer: bool = True,
) -> AgentResponse:

    with tracer.start_as_current_span("file_agent.run_react_agent") as span:
        span.set_attribute("file_agent.question", question)
        span.set_attribute("file_agent.tool_count", len(tools))

        tools_by_name = {tool.name: tool for tool in tools}
        tool_schemas = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt(history_summary)}
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})
        sources: list[SearchResult] = []

        iteration = 0
        for iteration in range(1, max_iterations + 1):
            if not _fits_context(llm_client, messages):
                logger.warning(
                    "Stopping tool-calling early at iteration %d/%d: accumulated "
                    "context is too full to safely continue",
                    iteration,
                    max_iterations,
                )
                break
            response = llm_client.generate_with_tools(messages, tool_schemas)
            log_generation(
                model=getattr(llm_client, "model", "unknown"),
                input_messages=messages,
                output=response.content,
                reasoning=response.reasoning,
                usage=response.usage,
            )

            if not response.tool_calls:
                answer = response.content or ""
                if verify_answer:
                    answer = _verify_answer(llm_client, question, answer, sources, tool_schemas)
                finish_trace(output=answer)
                span.set_attribute("file_agent.iterations", iteration)
                return AgentResponse(
                    answer=answer,
                    sources=sources,
                    iterations=iteration,
                    last_usage=response.usage,
                )

            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                result = _dispatch_tool(tools_by_name, call)
                sources.extend(result.sources)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.content}
                )

        final = llm_client.generate_with_tools(messages, tool_schemas, tool_choice="none")
        answer = final.content or "Unable to produce a final answer within the iteration limit."
        if verify_answer:
            answer = _verify_answer(llm_client, question, answer, sources, tool_schemas)
        finish_trace(output=answer)
        span.set_attribute("file_agent.iterations", iteration + 1)
        return AgentResponse(
            answer=answer,
            sources=sources,
            iterations=iteration + 1,
            last_usage=final.usage,
        )


def _assistant_message(response: ToolCallResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in response.tool_calls
        ],
    }


def _dispatch_tool(tools_by_name: dict[str, Tool], call: ToolCall) -> ToolResult:
    tool = tools_by_name.get(call.name)
    if tool is None:
        return ToolResult(content=f"Error: unknown tool '{call.name}'")

    try:
        result = tool.handler(**call.arguments)
    except Exception as exc:  # noqa: BLE001 - any tool failure becomes an observation, not a crash
        logger.warning("Tool %s failed: %s", call.name, exc, exc_info=True)
        result = ToolResult(content=f"Error running {call.name}: {exc}")

    log_tool_call(tool_name=call.name, arguments=call.arguments, output=result.content)
    return result
