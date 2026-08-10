import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse

_client: Langfuse | None = None


def _get_client() -> Langfuse | None:
    # Checked fresh on every call, not cached at import time: .env is loaded
    # lazily (inside create_generation_llm_client, well after this module is
    # first imported via the hf_cli -> agent.loop -> agent.observability
    # import chain), so freezing this at import time would miss credentials
    # that only exist in .env, not already in the shell environment.
    global _client
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
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
