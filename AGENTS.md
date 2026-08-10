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
- Optional VLM description of figures/diagrams in PDFs and picture shapes in
  PPTX slides (off by default, with graceful degradation when no VLM endpoint
  is reachable).
- Section-aware, token-budgeted chunking: blocks are grouped by heading, then
  whole sections are packed up to the retrieval encoder's token window (large
  tables split by rows, continuation chunks keep their heading as a breadcrumb,
  splits land on sentence boundaries), propagating section titles, page numbers
  and other metadata.
- Small-to-big retrieval: chunks are sized for the encoder, while each chunk
  carries its parent passage in `metadata["context"]`, which is what the QA
  prompt feeds to the LLM (deduplicated across chunks).
- Optional VLM figure description with a selectable backend (`VLM_BACKEND`:
  `off` / `smolvlm` local / `openai` endpoint) and a bounded per-document cost.
- In-memory LanceDB hybrid retrieval combining BM25 full-text search, semantic vector search, and reciprocal rank fusion (RRF).
- A QA prompt layer and end-to-end RAG orchestration.
- A ReAct tool-calling agent (`agent/loop.py`) that answers questions by
  reasoning and calling tools in a bounded loop (native OpenAI `tools=`, not
  prompt-embedded JSON): full-text/semantic search over indexed chunks, a
  restricted arithmetic evaluator, and — only when an `.xlsx` document is
  present — sandboxed pandas/openpyxl code execution in an isolated,
  network-disabled, resource-capped ephemeral Docker container per call. A
  hard iteration cap guarantees the loop always terminates with an answer.
- An `LLMClient` adapter built on the official OpenAI Python SDK, including
  native tool-calling (`generate_with_tools`).
- Yandex AI Studio and local OpenAI-compatible LLM backends.
- A Streamlit UI for multi-file upload, preview, search, and answer generation.
- Unified per-question tracing in the HF eval-dataset pipeline: one Langfuse
  trace covers parsing, chunking, indexing, and the ReAct agent loop (one
  generation per LLM turn, one span per tool call), bridged with the existing
  OpenTelemetry/Jaeger spans of the parsing/chunking/retrieval layers rather
  than duplicating instrumentation — both backends read the same shared OTel
  `TracerProvider` once `configure_telemetry()` has run (see `hf_cli.main()`).
- Pytest coverage for the main layers.

Supported extensions: `.md`, `.txt`, `.pdf`, `.docx`, `.html`, `.htm`, `.xlsx`, `.pptx`.

## Repository layout

```text
app.py                         # Streamlit UI
docker/sandbox.Dockerfile      # Image for the sandboxed spreadsheet tool
src/file_agent/
  document.py                 # Document and Block models
  pipeline.py                 # Parser selection by extension
  chunking.py                 # Document chunking
  retrieval.py                # Shared Retriever interface and SearchResult
  lancedb_retriever.py        # In-memory LanceDB hybrid retrieval
  qa.py                       # Context assembly and QA prompt
  rag.py                      # Document loading/chunking/indexing + single-shot RAG
  agent/                      # ReAct tool-calling agent
    loop.py                   # Agent loop: LLM turns, tool dispatch, iteration cap
    tools.py                  # Tool/ToolResult, search + calculator + spreadsheet tools
    sandbox.py                # Isolated Docker execution for the spreadsheet tool
    observability.py          # Langfuse trace/generation/span helpers
  parsers/                    # Supported file parsers
    docling_parser.py         # Structured PDF/DOCX parsing (Docling)
    routing.py                # Per-page OCR decision heuristics
    enhancer.py               # VLM description of figures/diagrams
  vlm/                        # VLM interface and OpenAI-compatible client
  utils/image_extractor.py    # PDF page-region crops and PPTX picture bytes for the VLM
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
- a standalone vector database or FAISS;
- Excel formula evaluation (the sandbox tool reads pandas/openpyxl cached
  values via `data_only=True` semantics, same as the XLSX parser — it does
  not evaluate live formulas).

"Complex agent architecture" was excluded here until the ReAct tool-calling
rewrite (`agent/`) explicitly lifted it, replacing the earlier Router/Planner
architecture — see `agent/loop.py` for the current scope (bounded tool-calling
loop, not open-ended planning/replanning).

"Image analysis for PPTX files" was excluded here until VLM figure description
was extended to picture shapes in `.pptx` slides (`parsers/pptx_parser.py`
emits `BlockType.IMAGE` blocks per picture shape; `parsers/enhancer.py`
extracts the shape's raw image bytes via `extract_image_from_pptx` — no
page rendering/cropping needed, unlike PDF).

OCR is implemented for PDF only (Docling with automatic per-page routing,
engine via `OCR_ENGINE`: `easyocr` default, reads Cyrillic + Latin, or
`rapidocr`; languages via `OCR_LANGS`, default `ru,en`). VLM figure/image
description covers both PDF (cropped page regions) and PPTX (picture shape
bytes) via the same OpenAI-compatible endpoint and `VLM_MAX_FIGURES`/
`VLM_MIN_FIGURE_AREA` cost controls.

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
- `LOCAL_LLM_MODEL`;
- `LOCAL_LLM_TEMPERATURE` (default `0.0`) — keep at 0: with vLLM's hermes
  tool-call parser, Qwen2.5 reliably emits a well-formed `<tool_call>` tag
  at temperature 0 but not higher (observed empirically); this only affects
  the ReAct agent's tool-calling reliability, not plain text generation.

ReAct agent:

- `AGENT_MAX_ITERATIONS` (default 6): hard cap on tool-calling turns per question.
- `SANDBOX_IMAGE` (default `file-agent-sandbox:latest`): image for the
  sandboxed spreadsheet tool, built once via
  `docker build -t file-agent-sandbox -f docker/sandbox.Dockerfile .`.
- vLLM must be started with `--enable-auto-tool-choice --tool-call-parser hermes`
  (see `docker-compose.yml`) for tool-calling to work against the local Qwen
  endpoint — verify the parser name/support against the deployed vLLM version
  before relying on it.

Langfuse (self-hosted, own stack — see `docker-compose.yml`'s
`file-agent-langfuse-*` services):

- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`;
- `LANGFUSE_POSTGRES_PASSWORD`, `LANGFUSE_CLICKHOUSE_PASSWORD`,
  `LANGFUSE_REDIS_PASSWORD`, `LANGFUSE_S3_ACCESS_KEY`, `LANGFUSE_S3_SECRET_KEY`,
  `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_NEXTAUTH_SECRET`
  (self-host infra secrets, only needed to run the `file-agent-langfuse-*`
  compose services, not by the application code itself).
- Tracing (`agent/observability.py`'s `pipeline_trace`) degrades to a silent
  no-op when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are unset (e.g. in
  tests).
- `pipeline_trace` opens its Langfuse trace around parsing/chunking/indexing/
  the agent loop for one dataset row (`hf_rag.process_qa_record`), not just
  the agent loop. It deliberately does not detach the ambient OTel context:
  when `configure_telemetry()` has already registered the global OTel
  `TracerProvider` (as `hf_cli.main()` does, before any parsing runs),
  Langfuse's client reuses that same provider instead of creating its own, so
  every existing `file_agent.telemetry` span created during the trace nests
  under it automatically — no extra instrumentation needed in the parsing/
  chunking/retrieval modules themselves.

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
- Langfuse;
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
