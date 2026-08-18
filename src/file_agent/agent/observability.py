import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from langfuse.span_filter import is_default_export_span
from opentelemetry.sdk.trace import ReadableSpan

_client: Langfuse | None = None


FILE_AGENT_INSTRUMENTATION_SCOPE = "file_agent"


def _export_file_agent_spans_too(span: ReadableSpan) -> bool:

    return is_default_export_span(span) or (
        span.instrumentation_scope is not None
        and span.instrumentation_scope.name == FILE_AGENT_INSTRUMENTATION_SCOPE
    )


def _get_client() -> Langfuse | None:
    global _client
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    if _client is None:
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3050"),
            should_export_span=_export_file_agent_spans_too,
        )
    return _client


@contextmanager
def pipeline_trace(question: str) -> Iterator[None]:
    client = _get_client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(
        name="process_qa_record", as_type="span", input=question
    ):
        yield


def log_generation(
    model: str,
    input_messages: list[dict[str, Any]],
    output: str | None,
    reasoning: str | None = None,
    usage: dict[str, int] | None = None,
) -> None:
    client = _get_client()
    if client is not None:
        client.start_observation(
            name="llm_turn",
            as_type="generation",
            model=model,
            input=input_messages,
            output=output,
            metadata={"reasoning": reasoning} if reasoning else None,
            usage_details=usage,
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
