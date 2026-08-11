# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

PDF and DOCX parsing uses Docling. OCR is enabled automatically for scanned PDF
pages. Optional VLM processing is disabled by default.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Configure the LLM backend in `.env`.

## Run the application

```bash
uv run streamlit run app.py
```

Local inference setup: [docs/local_inference.md](docs/local_inference.md).

## Tracing (OpenTelemetry + Jaeger)

The pipeline is instrumented with OpenTelemetry (`src/file_agent/telemetry.py`).
Spans are created in `parse_file`, `chunk_document`, `load_documents`,
`index_documents`, `LanceDBRetriever.index`/`.search`, `answer_question_with_context`
and `answer_indexed_documents`, and exported over OTLP/gRPC to Jaeger.

Run Jaeger locally:

```bash
docker compose up -d jaeger
```

The UI is available at http://localhost:16686 (service `file-agent-pipeline`).

Spans are sent to `localhost:4317` by default. Override the endpoint with the
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (see `.env.example`).

If Jaeger isn't running, `configure_telemetry()` still works — spans are sent
in the background via `BatchSpanProcessor` and simply won't arrive anywhere.

`generate_hf_dataset.py` (the HF eval-dataset pipeline) and `app.py` (the
Streamlit UI) both call `configure_telemetry()` before Langfuse's client ever
initializes. Langfuse then reuses the same global `TracerProvider` instead of
creating its own, so every span above (parsing, chunking, indexing,
retrieval) plus the ReAct agent loop's LLM turns and tool calls land in one
Langfuse trace per dataset row / per "Generate answer" click — see
`agent/observability.py`'s `pipeline_trace`.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
