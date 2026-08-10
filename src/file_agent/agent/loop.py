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

MAX_ITERATIONS_DEFAULT = 6

SYSTEM_PROMPT = (
    "You are a document question-answering agent. Use the available tools to "
    "find and compute the answer. Once you have enough information, respond "
    "directly with a concise final answer in plain text (no further tool "
    "calls), entirely in the same language as the question - do not switch "
    "languages mid-answer or mix in words from another language, even for "
    "technical terms. If the tools genuinely don't provide enough "
    "information, say so instead of guessing."
)


@dataclass
class AgentResponse:
    answer: str
    sources: list[SearchResult]
    iterations: int


def run_react_agent(
    question: str,
    llm_client: LLMClient,
    tools: list[Tool],
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        sources: list[SearchResult] = []

        for iteration in range(1, max_iterations + 1):
            response = llm_client.generate_with_tools(messages, tool_schemas)
            log_generation(
                model=getattr(llm_client, "model", "unknown"),
                input_messages=messages,
                output=response.content,
            )

            if not response.tool_calls:
                answer = response.content or ""
                finish_trace(output=answer)
                span.set_attribute("file_agent.iterations", iteration)
                return AgentResponse(answer=answer, sources=sources, iterations=iteration)

            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                result = _dispatch_tool(tools_by_name, call)
                sources.extend(result.sources)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.content}
                )

        # Hard cap reached without a final answer - force one from whatever
        # was gathered so the loop always terminates with *some* answer.
        final = llm_client.generate_with_tools(messages, tool_schemas, tool_choice="none")
        answer = final.content or "Unable to produce a final answer within the iteration limit."
        finish_trace(output=answer)
        span.set_attribute("file_agent.iterations", max_iterations + 1)
        return AgentResponse(answer=answer, sources=sources, iterations=max_iterations + 1)


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
