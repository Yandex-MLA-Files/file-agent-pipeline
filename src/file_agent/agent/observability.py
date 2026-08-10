import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from langfuse.span_filter import is_default_export_span
from opentelemetry.sdk.trace import ReadableSpan

_client: Langfuse | None = None

# file_agent.telemetry's tracer (file_agent.telemetry.tracer = trace.get_tracer("file_agent"))
FILE_AGENT_INSTRUMENTATION_SCOPE = "file_agent"


def _export_file_agent_spans_too(span: ReadableSpan) -> bool:
    """LangfuseSpanProcessor's default should_export_span is an allowlist -
    it only forwards spans from Langfuse's own tracer, spans carrying a
    gen_ai.* attribute, or a fixed set of known LLM-instrumentation scope
    names (langfuse._client.span_filter.KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES).
    file_agent.telemetry's spans (parse_file, docling_parse, chunk_document,
    retriever_index/search, run_react_agent, ...) match none of those, so
    without this override they're silently dropped even though they reach
    the shared TracerProvider fine - only the Langfuse-native
    process_qa_record/llm_turn/tool:* spans would show up in a trace.
    """
    return is_default_export_span(span) or (
        span.instrumentation_scope is not None
        and span.instrumentation_scope.name == FILE_AGENT_INSTRUMENTATION_SCOPE
    )


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
            should_export_span=_export_file_agent_spans_too,
        )
    return _client


@contextmanager
def pipeline_trace(question: str) -> Iterator[None]:
    """Open one Langfuse trace (root span) for one dataset row's full pipeline
    run: parsing, chunking, indexing, and the ReAct agent loop.

    Silently no-ops when LANGFUSE_PUBLIC_KEY/SECRET_KEY aren't set, so callers
    and their tests never need Langfuse to be reachable.

    Deliberately does NOT detach the ambient OTel context before opening this
    span. Langfuse's own tracer is OTel-based; when file_agent.telemetry's
    configure_telemetry() has already registered a real (non-proxy) global
    TracerProvider, Langfuse's client reuses that same provider instead of
    creating its own (langfuse._client.resource_manager._init_tracer_provider)
    and simply adds its own span processor to it. That makes every existing
    file_agent.telemetry span created inside this trace (docling_parse,
    chunk_document, retriever_index/search, llm_generate*, ...) show up as a
    nested child here automatically, with no extra instrumentation in those
    modules - the whole point of opening the trace at this level rather than
    only around the agent loop. Callers that never call configure_telemetry()
    (e.g. tests, or a future caller without it) are unaffected: Langfuse then
    creates its own provider and this trace is simply its own root, same as
    before.
    """
    client = _get_client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(
        name="process_qa_record", as_type="span", input=question
    ):
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
