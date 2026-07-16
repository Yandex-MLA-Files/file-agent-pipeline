# File Agent Pipeline

A lightweight Python RAG application for answering questions about user-provided files. It parses documents, splits extracted text into chunks, retrieves relevant context with LanceDB hybrid search, and generates source-grounded answers through an OpenAI-compatible LLM endpoint.

Supported formats: Markdown, PDF, HTML, XLSX, and PPTX.

## Quick start

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

Copy `.env.example` to `.env` and configure either Yandex AI Studio or a local OpenAI-compatible endpoint before generating answers. See [local inference setup](docs/local_inference.md) for local vLLM and SGLang examples.

## Retrieval

Uploaded document chunks are indexed in an in-memory LanceDB table. Retrieval combines BM25 full-text search with semantic vector search and uses reciprocal rank fusion (RRF) to produce the final ranking. The embedding model runs locally through `sentence-transformers`; the first run may download its model files.

The index is isolated to the current Streamlit session and is cleared when the session or application process ends. No separate database server or Docker container is required.

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
