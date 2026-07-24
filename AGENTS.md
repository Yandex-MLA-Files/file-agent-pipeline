# AGENTS.md

## Project overview

`file-agent-pipeline` is a Python ML project that implements a small RAG pipeline over user-provided files:

```text
files
  -> parsing
  -> Document / Block
  -> chunking
  -> retrieval
  -> QA prompt
  -> LLM
  -> answer and sources
```

Users can upload one or more documents, preview extracted text, find relevant chunks, and generate an LLM answer with source metadata.

## Current capabilities

- A shared `Document` / `Block` representation with optional structural
  annotations (`block_type`, `page_number`, `bbox`, `vlm_description`) and a
  `Document.to_markdown()` export.
- Structured PDF and DOCX parsing via Docling (reading order, headings, tables,
  figures, formulas), plus Markdown, HTML, XLSX, and PPTX parsers.
- Automatic per-page OCR routing for PDFs (`parsers/routing.py`): OCR is enabled
  only for scanned/image pages, decided locally with no network calls.
- Optional VLM description of figures/diagrams in PDFs (off by default, with
  graceful degradation when no VLM endpoint is reachable).
- Section-aware chunking: blocks are grouped by heading, then whole sections are
  packed into retrieval-sized passages (large tables split by rows with a
  repeated header, continuation chunks keep their heading as a breadcrumb),
  propagating section titles, page numbers and other metadata.
- In-memory LanceDB hybrid retrieval combining BM25 full-text search, semantic vector search, and reciprocal rank fusion (RRF).
- A QA prompt layer and end-to-end RAG orchestration.
- An `LLMClient` adapter built on the official OpenAI Python SDK.
- Yandex AI Studio and local OpenAI-compatible LLM backends.
- A Streamlit UI for multi-file upload, preview, search, and answer generation.
- Pytest coverage for the main layers.

Supported extensions: `.md`, `.pdf`, `.docx`, `.html`, `.htm`, `.xlsx`, `.pptx`.

## Repository layout

```text
app.py                         # Streamlit UI
src/file_agent/
  document.py                 # Document and Block models
  pipeline.py                 # Parser selection by extension
  chunking.py                 # Document chunking
  retrieval.py                # Shared Retriever interface and SearchResult
  lancedb_retriever.py        # In-memory LanceDB hybrid retrieval
  qa.py                       # Context assembly and QA prompt
  rag.py                      # End-to-end RAG orchestration
  parsers/                    # Supported file parsers
    docling_parser.py         # Structured PDF/DOCX parsing (Docling)
    routing.py                # Per-page OCR decision heuristics
    enhancer.py               # VLM description of figures/diagrams
  vlm/                        # VLM interface and OpenAI-compatible client
  utils/image_extractor.py    # Crop PDF page regions to images for the VLM
  llm/                        # LLM interface, adapter, and factory
tests/                        # Pytest suite
docs/local_inference.md       # Local LLM endpoint setup
```

## Architecture rules

- Keep parsing, retrieval, QA, and LLM integration as separate layers.
- Access retrieval through the `Retriever` interface and keep LanceDB-specific code in `lancedb_retriever.py`.
- Every parser must return the shared `Document` representation containing `Block` objects.
- The first four `Block` fields (`id`, `text`, `type`, `metadata`) are a stable,
  backward-compatible interface; the structural fields (`block_type`,
  `page_number`, `bbox`, `vlm_description`) are optional and default to `None`.
- Preserve available source metadata, including:
  - `page_number` and `bbox` for PDF;
  - `slide_number` for PPTX;
  - `sheet_name` for XLSX;
  - `block_type` for the block kind;
  - `table_of_contents` and `page_analysis` on `Document.metadata`;
  - the source file name and other useful source coordinates.
- Keep parser selection by extension in `src/file_agent/pipeline.py`.
- Keep parsing offline by default: OCR is auto-routed locally and the VLM is
  opt-in, so `parse_file(path)` must never require a network service.
- Access LLMs only through the `LLMClient` interface.
- Keep backend-specific configuration in `src/file_agent/llm/factory.py` and environment variables.
- Avoid complex abstractions without a practical need. Prefer simple, readable code with type hints.
- Add tests for new parsers and new behavior.
- Tests must not call real cloud or local LLM endpoints. Mock or fake all network interactions.

## Current-stage exclusions

Do not add the following without a separate task:

- LangChain or LangGraph;
- complex agent architecture;
- a standalone vector database or FAISS;
- image analysis for PPTX files;
- Excel formula evaluation.

OCR and VLM support are implemented for PDF only: OCR via Docling with automatic
per-page routing (engine via `OCR_ENGINE`: `easyocr` default, reads Cyrillic +
Latin, or `rapidocr`; languages via `OCR_LANGS`, default `ru,en`), and VLM figure
description via an OpenAI-compatible endpoint.

The XLSX parser uses `data_only=True`: it reads cached formula values but does not calculate formulas.

## LLM configuration

Never hardcode secrets or identifiers. Do not commit a real `.env` file. Update `.env.example` whenever configuration changes.

Shared setting:

- `LLM_BACKEND`: `yandex` or `local`.

Yandex AI Studio:

- `YANDEX_API_KEY`;
- `YANDEX_FOLDER_ID`;
- `YANDEX_MODEL`;
- `YANDEX_BASE_URL`.

Local OpenAI-compatible backend:

- `LOCAL_LLM_BASE_URL`;
- `LOCAL_LLM_API_KEY`;
- `LOCAL_LLM_MODEL`.

Never make real API requests in tests or add working credentials to code, fixtures, logs, or documentation.

## Stack

- Python 3.11+;
- Streamlit;
- PyMuPDF;
- BeautifulSoup;
- openpyxl;
- python-pptx;
- sentence-transformers;
- LanceDB;
- openai;
- pytest.

## Change guidelines

Before editing, inspect the relevant module and existing tests. Preserve backward compatibility unless the task explicitly requires a behavior change.

After editing:

- add or update tests for changed behavior;
- run at least the relevant tests;
- run the full test suite when practical;
- update `README.md`, `.env.example`, or `docs/` when interfaces, configuration, supported formats, or run commands change.

Do not commit temporary files, caches, models, user documents, `.env`, or other secrets.

## Commands

Install dependencies:

```powershell
uv sync
```

Run all tests:

```powershell
uv run pytest
```

Check linting and formatting:

```powershell
uv run ruff check .
uv run ruff format --check .
```

Run Streamlit:

```powershell
uv run streamlit run app.py
```

See `docs/local_inference.md` for local inference setup.
