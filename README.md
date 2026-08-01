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
RAG_MODE=standard  # retrieve -> generate
RAG_MODE=agentic   # analyze -> retrieve -> grade -> rewrite/retry -> generate
RAG_MAX_RETRIES=2  # maximum query rewrites in agentic mode
```

Agentic retries are bounded. If the question needs clarification or no relevant
context is found after the configured attempts, the graph stops without calling
the answer-generation node. Query analysis, context grading, and every rewrite
are additional LLM calls, so `agentic` mode trades latency and cost for better
retrieval recovery.

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

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
