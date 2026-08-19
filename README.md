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
- inspect exact document properties and full-document text statistics, including
  PDF page count, file size, native title/author metadata, and word counts from the
  native PDF text layer with parsed/OCR fallback for image-only pages;
- exhaustively count a literal word or phrase across every parsed block and return
  all matching page numbers without estimating from retrieval top-k;
- read a complete document sequentially in bounded pages for exhaustive analysis
  or unstructured files;
- read the surrounding context of a previously found chunk;
- read a section, including its nested subsections;
- read a PDF/DOCX page, PPTX slide, or XLSX sheet;
- read table rows in bounded pages;
- calculate decimal sums, differences, ratios, averages, shares, and percentage
  changes over values found in the documents;
- discover figures in a PDF and analyze a selected figure crop or full page with
  the configured VLM.

The Hugging Face batch generator accepts `--rag-mode standard` or
`--rag-mode tool_agent`. Batch tool-agent runs are stateless between dataset rows,
require a document evidence tool before accepting an answer, and export the exact
evidence shown to the model together with tool-call, token, and duration diagnostics.
See [docs/hf_dataset_generation.md](docs/hf_dataset_generation.md) for the paired-run
workflow.

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

The Streamlit UI indexes an unchanged set of uploaded documents only once. Its
sidebar can create and switch between chats; every chat has a separate
`thread_id` and keeps its visible messages without re-indexing the documents.
Uploading a different file set clears the old chat list, starts a fresh chat,
and performs ingestion for the new set. Answer sources are shown as compact
file/page references instead of internal retrieval and tool diagnostics. The
current checkpointer is process-local; it can later be replaced by a SQLite or
PostgreSQL checkpointer without changing the graph nodes.

For on-demand visual questions, the tool agent uses
`analyze_document_visual`. The tool accepts only an indexed `source_file` and
either a `visual_id` returned by `get_document_outline` or a PDF `page_number`;
it cannot access arbitrary paths, URLs, or bounding boxes. Original uploaded
file bytes stay in a process-local asset store in the Streamlit session and are
passed through LangGraph runtime context, never through graph state or chat
history. The selected PDF region is sent to the VLM, and only the bounded text
analysis and source coordinates are returned as a tool observation.

The existing ingestion-time figure descriptions remain useful for retrieval and
visual discovery. The on-demand tool performs a fresh, question-specific visual
analysis before the agent makes claims about a chart or diagram. Visual tool
analysis currently supports PDF files; other document tools and standard text
RAG behavior are unchanged.

`VLM_MAX_VISUAL_PIXELS` bounds the rendered image sent by the on-demand visual
tool (default `1500000`); a higher value can preserve small labels at the cost of
larger requests. `VLM_MAX_RETRIES` configures retries for transient failures of an
OpenAI-compatible VLM endpoint.

The LLM and VLM can point at the same multimodal vLLM deployment. For example:

```text
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen3.5-27B
LOCAL_LLM_ENABLE_THINKING=false

VLM_BACKEND=openai
VLM_BASE_URL=http://localhost:8000/v1
VLM_MODEL=Qwen/Qwen3.5-27B
VLM_ENABLE_THINKING=false
```

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
