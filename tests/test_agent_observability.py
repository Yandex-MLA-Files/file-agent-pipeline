from types import SimpleNamespace

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace import TracerProvider

from file_agent.agent import observability


class FakeObservation:
    def __init__(self):
        self.ended = False

    def end(self):
        self.ended = True


class FakeSpanContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeLangfuse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observations = []
        self.updated_span_output = None

    def start_as_current_observation(self, **kwargs):
        return FakeSpanContext()

    def start_observation(self, **kwargs):
        self.observations.append(kwargs)
        return FakeObservation()

    def update_current_span(self, output):
        self.updated_span_output = output


def _reset_client(monkeypatch):
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(observability, "Langfuse", FakeLangfuse)


def test_get_client_is_none_without_credentials(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability._get_client() is None


def test_get_client_is_none_with_only_one_credential(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability._get_client() is None


def test_get_client_is_not_frozen_at_import_time(monkeypatch):
    """Regression test: credentials set only after this module was already
    imported (e.g. loaded from .env later in the process) must still work -
    the enabled check must be re-evaluated on every call, not cached once."""
    _reset_client(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert observability._get_client() is None

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    client = observability._get_client()
    assert isinstance(client, FakeLangfuse)


def test_get_client_reuses_the_same_instance(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    first = observability._get_client()
    second = observability._get_client()

    assert first is second


def test_get_client_exports_file_agent_spans_alongside_langfuse_defaults(monkeypatch):
    """Regression test: LangfuseSpanProcessor's default should_export_span is
    an allowlist (Langfuse's own tracer, gen_ai.* spans, known LLM
    instrumentation scopes) that drops everything else - including
    file_agent.telemetry's parse_file/chunk_document/retriever_* spans,
    which are the whole point of bridging onto the shared TracerProvider.
    _get_client() must override it to also allow the file_agent scope."""
    _reset_client(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    client = observability._get_client()
    should_export_span = client.kwargs["should_export_span"]

    file_agent_span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="file_agent"), attributes=None
    )
    unrelated_span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="some_other_library"), attributes=None
    )

    assert should_export_span(file_agent_span) is True
    assert should_export_span(unrelated_span) is False


def test_pipeline_trace_is_a_noop_without_credentials(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with observability.pipeline_trace("question"):
        pass  # must not raise even though no client is configured

    observability.log_generation(model="m", input_messages=[], output="a")
    observability.log_tool_call(tool_name="t", arguments={}, output="o")
    observability.finish_trace(output="done")


def test_log_generation_and_tool_call_and_finish_trace_reach_the_client(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    with observability.pipeline_trace("question"):
        observability.log_generation(model="m", input_messages=[{"role": "user"}], output="a")
        observability.log_tool_call(tool_name="search", arguments={"query": "x"}, output="o")
        observability.finish_trace(output="final answer")

    client = observability._get_client()
    assert [obs["as_type"] for obs in client.observations] == ["generation", "tool"]
    assert client.updated_span_output == "final answer"


def test_pipeline_trace_does_not_detach_the_ambient_otel_context(monkeypatch):
    """Regression test: pipeline_trace must NOT clear the ambient OTel context
    before opening its own Langfuse span. The whole point of opening the
    Langfuse trace at the process_qa_record level is that, once
    configure_telemetry() has registered a shared TracerProvider, Langfuse
    reuses it - so file_agent.telemetry spans created inside this trace
    (docling_parse, chunk_document, retriever_index/search, ...) nest under
    it automatically. Detaching context here (as an earlier, narrower fix
    once did for the agent-loop-only trace) would break that nesting."""
    _reset_client(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    seen_span_during_trace = []

    class SpyLangfuse(FakeLangfuse):
        def start_as_current_observation(self, **kwargs):
            seen_span_during_trace.append(otel_trace_api.get_current_span())
            return super().start_as_current_observation(**kwargs)

    monkeypatch.setattr(observability, "Langfuse", SpyLangfuse)

    ambient_tracer = TracerProvider().get_tracer("file_agent")
    with ambient_tracer.start_as_current_span("file_agent.load_documents") as ambient_span:
        with observability.pipeline_trace("question"):
            pass

    assert seen_span_during_trace[0] is ambient_span
