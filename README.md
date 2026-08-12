# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

PDF and DOCX parsing uses Docling. OCR is enabled automatically for scanned PDF
pages. Optional VLM processing is disabled by default.

LangGraph orchestrates the ingestion, standard QA, and tool-agent workflows:

```text
ingestion: files/documents -> parsing -> chunking -> indexing
QA:        question -> retrieval -> answer generation -> response
agent:     conversation -> model -> document tools -> model -> response
```

Keeping ingestion separate lets the Streamlit application index uploaded files
once and reuse the same in-memory LanceDB retriever for multiple questions.

The QA workflow has two modes configured in `.env`:

```text
RAG_MODE=standard    # retrieve -> generate
RAG_MODE=tool_agent  # model -> document tools -> model -> final answer
RAG_MAX_TOOL_ROUNDS=4
RAG_HISTORY_TURNS=6  # completed user/assistant pairs kept in memory
```

The tool agent can search the index multiple times, optionally restrict search to
one source file, and navigate uploaded documents without accessing arbitrary
filesystem paths. Its read-only tools can:

- list documents and inspect their sections and tables;
- read the surrounding context of a previously found chunk;
- read a section, including its nested subsections;
- read a PDF/DOCX page, PPTX slide, or XLSX sheet;
- read table rows in bounded pages.

Long sections and locations return `next_offset`; tables use a bounded row
`offset` and `limit`. Direct reads are preserved as answer sources just like
retrieval results. All tools receive the active retriever and parsed documents
through LangGraph runtime context. Tool execution is bounded by
`RAG_MAX_TOOL_ROUNDS`; after the limit, the model must answer from the
observations already collected.

The tool-agent graph is compiled with a LangGraph in-memory checkpointer. The
Streamlit session keeps a stable `thread_id`, so follow-up questions can use the
last `RAG_HISTORY_TURNS` completed user/assistant pairs to resolve references such
as "and in the second quarter?" or "what are the penalties there?". Tool calls
and tool observations stay available during the active turn, but are removed
from the long-lived checkpoint after the final answer. Conversation history is
context, not evidence: each new question must call a document tool before making
new factual claims about the uploaded files.

The Streamlit UI indexes an unchanged set of uploaded documents only once. **New
chat** creates a fresh `thread_id` and clears the visible messages without
re-indexing those documents. Uploading a different file set automatically starts
a new chat and performs ingestion for the new set. The current checkpointer is
process-local; it can later be replaced by a SQLite or PostgreSQL checkpointer
without changing the graph nodes.

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
