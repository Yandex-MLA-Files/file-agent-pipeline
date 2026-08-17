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
- Format-aware structured parsers (`PARSER_PROFILE=structured`, default) that
  emit typed blocks (headings with levels, paragraphs, lists, Markdown tables,
  figures with image bytes, code, formulas) for every format: PDF via Docling
  (with heading-level inference, list grouping, caption attachment), DOCX via
  python-docx (paragraph-faithful; Docling as fallback), PPTX (slide titles,
  reading order, tables, charts, notes), XLSX (Markdown tables per data region
  with header detection), HTML, Markdown and plain text (prose reflow,
  transcript timestamps, encoding detection). The original flat parsers remain
  available as `PARSER_PROFILE=legacy` (`parsers/legacy/`).
- Automatic per-page OCR routing for PDFs (`parsers/routing.py`): OCR runs
  only for scanned/image pages, decided locally with no network calls. The
  default engine (`OCR_ENGINE=vlm`) transcribes those pages with the
  multimodal chat model (`parsers/vlm_ocr.py`) and falls back to EasyOCR when
  no VLM endpoint is configured; `easyocr`/`rapidocr` run inside Docling.
- VLM description of figures in every format through a selectable backend
  (`VLM_BACKEND`: `llm` default — the answering model's own multimodal
  endpoint —, `openai`, `smolvlm`, `off`) with a bounded per-document cost;
  parsing degrades gracefully when no endpoint is reachable.
- Structured chunking (`CHUNKING_STRATEGY=structured`, default): blocks are
  grouped into leaf sections with their full heading path; whole sections are
  packed up to the encoder's token budget (`CHUNK_TARGET_TOKENS`, 384 by
  default); every chunk is prefixed with a "Title > Chapter > Section"
  breadcrumb and carries `heading_path`; prose splits on sentences, lists on
  items, tables on rows with the header repeated, code on lines. The original
  chunker is `CHUNKING_STRATEGY=legacy` (`chunking_legacy.py`).
- Small-to-big retrieval: chunks are sized for the encoder, while each chunk
  carries a parent passage window in `metadata["context"]` (the surrounding
  section, opened by its heading path), which is what the QA prompt feeds to
  the LLM (deduplicated across chunks).
- Dense retrieval with `BAAI/bge-m3` by default (`EMBEDDING_MODEL` to change).
- In-memory LanceDB hybrid retrieval combining BM25 full-text search, semantic vector search, and reciprocal rank fusion (RRF).
- A QA prompt layer (grounded, complete answers with compact source headers;
  `QA_PROMPT=v1` keeps the original prompt) and end-to-end RAG orchestration.
- A multi-step document agent (`src/file_agent/agent/`): a think-act-observe
  loop where the LLM plans tool calls (`search_documents`, `list_documents`,
  `read_section`) over the indexed documents; tool calls are JSON parsed
  client-side, so any OpenAI-compatible backend works without server-side
  tool-call support. Bounded session memory (`AgentSession`) enables
  follow-up questions (see `docs/agent.md`).
- An `LLMClient` adapter built on the official OpenAI Python SDK, plus a
  `chat(messages)` method for multi-turn conversations.
- Yandex AI Studio and local OpenAI-compatible LLM backends.
- A Streamlit UI for multi-file upload, preview, search, and answer generation.
- Pytest coverage for the main layers.

Supported extensions: `.md`, `.txt`, `.pdf`, `.docx`, `.html`, `.htm`, `.xlsx`, `.pptx`.

## Repository layout

```text
app.py                         # Streamlit UI
src/file_agent/
  document.py                 # Document and Block models
  pipeline.py                 # Parser selection by extension, OCR/VLM policy
  chunking.py                 # Structured chunker (default strategy)
  chunking_legacy.py          # Original chunker (CHUNKING_STRATEGY=legacy)
  retrieval.py                # Shared Retriever interface and SearchResult
  lancedb_retriever.py        # In-memory LanceDB hybrid retrieval
  qa.py                       # Context assembly and QA prompt
  rag.py                      # End-to-end RAG orchestration
  parsers/                    # Supported file parsers
    common.py                 # Shared helpers (Markdown tables, heading levels, encodings)
    docling_parser.py         # Structured PDF parsing (Docling) + block post-processing
    docx_parser.py            # DOCX via python-docx
    pptx_parser.py            # PPTX via python-pptx
    xlsx_parser.py            # XLSX via openpyxl (tables per region)
    html_parser.py, md_parser.py, txt_parser.py
    legacy/                   # Original flat parsers (PARSER_PROFILE=legacy)
    routing.py                # Per-page OCR decision heuristics
    vlm_ocr.py                # VLM transcription of scanned pages
    enhancer.py               # VLM description of figures/diagrams
  vlm/                        # VLM interface and OpenAI-compatible client
  utils/image_extractor.py    # Crop PDF page regions to images for the VLM
  llm/                        # LLM interface, adapter, and factory
tests/                        # Pytest suite
docs/local_inference.md       # Local LLM endpoint setup
docs/agent.md                 # Document agent design and serving notes
docs/parsing_and_chunking.md  # Parsing/chunking design and evaluation results
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
- Keep parsing usable offline: OCR routing is local, and when no VLM endpoint
  is configured (`VLM_BACKEND=off` or no `LOCAL_LLM_BASE_URL`) figure
  description is skipped and page OCR falls back to EasyOCR, so
  `parse_file(path)` never *requires* a network service.
- Keep chunk sizes aligned with the retrieval encoder's token window. Anything
  longer is silently truncated when embedded, so budget chunks with the encoder's
  tokenizer (see `get_embedding_tokenizer`) instead of raw character counts, and
  revisit the budget whenever the embedding model changes.
- Keep what is embedded and what the LLM reads separate: chunk text is the
  retrieval unit, `metadata["context"]` is the answer unit. Anything that widens
  the answer context belongs in the parent passage, not in the chunk text.
- Access VLMs only through `file_agent.vlm.factory.create_vlm_client()`, keep the
  backend choice in environment variables, and keep figure description bounded
  (largest figures first, tiny decorative images skipped) so cost stays
  predictable.
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
- Excel formula evaluation.

OCR is implemented for PDF only (per-page routing; `OCR_ENGINE`: `vlm` default,
`easyocr`, `rapidocr`; `OCR_LANGS` for EasyOCR, default `ru,en`). VLM figure
description works for PDF (page crops) and for DOCX/PPTX (embedded images).

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
