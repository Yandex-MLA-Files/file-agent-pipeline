import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse

_client: Langfuse | None = None
_enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def _get_client() -> Langfuse | None:
    global _client
    if not _enabled:
        return None
    if _client is None:
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3050"),
        )
    return _client


@contextmanager
def agent_trace(question: str) -> Iterator[None]:
    """Open one Langfuse trace (root span) for a single agent run.

    Silently no-ops when LANGFUSE_PUBLIC_KEY/SECRET_KEY aren't set, so the
    agent loop and its tests never need Langfuse to be reachable. This traces
    LLM turns and tool calls only - parsing/chunking/retrieval keep the
    existing OpenTelemetry/Jaeger spans (file_agent.telemetry) unchanged.
    """
    client = _get_client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(name="react_agent", as_type="span", input=question):
        yield


def log_generation(model: str, input_messages: list[dict[str, Any]], output: str | None) -> None:
    client = _get_client()
    if client is not None:
        client.start_observation(
            name="llm_turn", as_type="generation", model=model, input=input_messages, output=output
        ).end()


def log_tool_call(tool_name: str, arguments: dict[str, Any], output: str) -> None:
    client = _get_client()
    if client is not None:
        client.start_observation(
            name=f"tool:{tool_name}", as_type="tool", input=arguments, output=output
        ).end()


def finish_trace(output: str) -> None:
    client = _get_client()
    if client is not None:
        client.update_current_span(output=output)
