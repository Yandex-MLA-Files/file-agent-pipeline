# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

PDF and DOCX parsing uses Docling. OCR is enabled automatically for scanned PDF
pages. Optional VLM processing is disabled by default.

LangGraph orchestrates two independent workflows:

```text
ingestion: files/documents -> parsing -> chunking -> indexing
QA:        question -> retrieval -> answer generation -> response
```

Keeping ingestion separate lets the Streamlit application index uploaded files
once and reuse the same in-memory LanceDB retriever for multiple questions.

The QA workflow has two modes configured in `.env`:

```text
RAG_MODE=standard    # retrieve -> generate
RAG_MODE=tool_agent  # model -> document tools -> model -> final answer
RAG_MAX_TOOL_ROUNDS=4
```

The tool agent can search the index multiple times, list uploaded documents, and
inspect a document's table of contents. All tools are read-only and receive the
active retriever and documents through LangGraph runtime context. Tool execution
is bounded by `RAG_MAX_TOOL_ROUNDS`; after the limit, the model must answer from
the observations already collected.

`tool_agent` requires an OpenAI-compatible model and endpoint with native tool
calling support. The standard mode continues to work with text-generation-only
models.

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

## Tracing (OpenTelemetry + Jaeger/Langfuse)

The pipeline is instrumented with OpenTelemetry (`src/file_agent/telemetry.py`).
Spans are created in `parse_file`, `chunk_document`, `load_documents`,
`index_documents`, `LanceDBRetriever.index`/`.search`, `answer_question_with_context`
and `answer_indexed_documents`. Jaeger export uses OTLP/gRPC. When Langfuse
credentials are configured, the same trace is also exported via OTLP/HTTP with
Langfuse observation types and LLM input/output, model, and token usage.

Run Jaeger locally:

```bash
docker compose up -d jaeger
```

The UI is available at http://localhost:16686 (service `file-agent-pipeline`).

Spans are sent to `localhost:4317` by default. Override the endpoint with the
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (see `.env.example`).

If Jaeger isn't running, `configure_telemetry()` still works — spans are sent
in the background via `BatchSpanProcessor` and simply won't arrive anywhere.

To send traces to a local Langfuse project, create API keys in the Langfuse
project settings and add these values to `.env`:

```text
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

Restart Streamlit, upload a document, and generate an answer. The first trace
then appears in Langfuse under **Tracing**. Langfuse receives prompt and answer
text, so do not enable this export for documents that must not leave the app's
observability boundary.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
