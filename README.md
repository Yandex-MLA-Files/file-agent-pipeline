# File Agent Pipeline

A Python RAG application for answering questions about user-provided files. It parses documents, splits extracted text into chunks, retrieves relevant context with BM25 and semantic search, and generates source-grounded answers through an OpenAI-compatible LLM endpoint.

Supported formats: Markdown, PDF, HTML, XLSX, and PPTX.

## Quick start

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

Copy `.env.example` to `.env` and configure either Yandex AI Studio or a local OpenAI-compatible endpoint before generating answers. See [local inference setup](docs/local_inference.md) for local vLLM and SGLang examples.

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
