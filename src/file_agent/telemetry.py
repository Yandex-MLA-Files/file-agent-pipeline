import logging
import os
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

_configured = False

LOG_DIR = Path("logs")
COST_LOGGER_NAME = "file_agent.cost"
COST_LOG_PATH = LOG_DIR / "llm_costs.csv"
COST_LOG_HEADER = "timestamp,model,prompt_tokens,completion_tokens,cost_rub"


def configure_telemetry(service_name: str = "file-agent-pipeline") -> None:
    global _configured
    if _configured:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "app.log"),
        ],
    )
    _configure_cost_logger()

    _configured = True


def _configure_cost_logger() -> None:
    is_new_file = not COST_LOG_PATH.exists() or COST_LOG_PATH.stat().st_size == 0

    cost_logger = logging.getLogger(COST_LOGGER_NAME)
    cost_logger.setLevel(logging.INFO)
    cost_logger.propagate = False

    handler = logging.FileHandler(COST_LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    cost_logger.addHandler(handler)

    if is_new_file:
        cost_logger.info(COST_LOG_HEADER)


tracer = trace.get_tracer("file_agent")

SpanIdentity = tuple[int, int]


def span_identity(span: trace.Span) -> SpanIdentity:
    span_context = span.get_span_context()
    return span_context.trace_id, span_context.span_id


@contextmanager
def resume_span(identity: SpanIdentity | None):
    if identity is None:
        yield
        return

    trace_id, span_id = identity
    parent = NonRecordingSpan(
        SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    token = otel_context.attach(trace.set_span_in_context(parent))
    try:
        yield
    finally:
        otel_context.detach(token)
