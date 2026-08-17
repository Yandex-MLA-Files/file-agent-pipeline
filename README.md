# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

Two answer modes are available: single-pass RAG (one retrieval, one LLM
call) and a multi-step agent that plans its own tool calls — search,
document overview, reading whole sections — before answering. See
[docs/agent.md](docs/agent.md).

Every format is parsed into typed blocks (headings with levels, paragraphs,
lists, Markdown tables, figures, code): PDF via Docling, DOCX via python-docx,
PPTX/XLSX/HTML/Markdown/TXT with dedicated structure-aware parsers. Chunks are
section-coherent, prefixed with their heading path and budgeted in the
retrieval encoder's tokens (BGE-M3 by default). Scanned PDF pages are detected
per page and transcribed by the multimodal chat model (Qwen3.5); figures in
PDF/DOCX/PPTX are described by the same model. Design and evaluation results:
[docs/parsing_and_chunking.md](docs/parsing_and_chunking.md).

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

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
