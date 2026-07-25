# File Agent Pipeline

A Python RAG application for answering questions about user-provided files. It parses documents, splits extracted text into chunks, retrieves relevant context with BM25 and semantic search, and generates source-grounded answers through an OpenAI-compatible LLM endpoint.

Supported formats: Markdown, PDF, DOCX, HTML, XLSX, and PPTX.

## Document parsing

PDF and DOCX are parsed with [Docling](https://github.com/DS4SD/docling), which
reconstructs reading order and classifies each element (heading, table, figure,
formula). This produces a rich `Document` with per-block page numbers and
bounding boxes, a generated table of contents, and a uniform Markdown export via
`Document.to_markdown()`.

- **OCR is decided automatically.** Before parsing, each PDF page is analyzed
  locally (text density and image coverage) to decide whether it needs OCR, so
  born-digital pages stay fast and only scanned/image pages are OCR'd. OCR uses
  EasyOCR by default (reads Cyrillic and Latin — documents are often Russian;
  languages via `OCR_LANGS`, default `ru,en`); set `OCR_ENGINE=rapidocr` for a
  faster offline Latin-only engine. Override the decision with
  `parse_file(path, enable_ocr="on" | "off")`.
- **Figures can be described by a VLM.** Figures and diagrams are cropped and
  described, and the description is folded into the searchable text. Off by
  default; pick a backend with `VLM_BACKEND`:
  - `smolvlm` — local SmolVLM-256M through transformers: no server, no API key,
    ~500 MB one-time download, free. Cheapest working option, but a 256M model
    reads only simple figures reliably; point `VLM_LOCAL_MODEL` at a larger
    SmolVLM checkpoint for better captions.
  - `openai` — any OpenAI-compatible vision endpoint (`VLM_BASE_URL` /
    `VLM_MODEL`), e.g. a local Ollama `qwen2.5-vl:7b` (free, needs ~6 GB RAM) or
    a hosted API. Use this when figure content actually matters.

  Cost is bounded either way: at most `VLM_MAX_FIGURES` figures per document
  (largest first) and tiny decorative images are skipped.

If Docling cannot process a PDF, the pipeline falls back to a plain PyMuPDF text
extraction so parsing never hard-fails.

## Chunking

Structured parsing yields many small blocks, so chunking works in two levels to
avoid both extremes — one tiny chunk per block, and one giant chunk that mixes
unrelated sections:

1. blocks are grouped into **sections** (a heading plus its body), so a heading
   always opens a chunk and never dangles at the end of the previous one;
2. whole sections are **packed together up to a size budget** (small adjacent
   sections merge), oversized sections are split block by block, small tables are
   kept whole while large ones are split by rows (repeating the header), and each
   continuation chunk keeps its section heading as a breadcrumb.

The budget is measured in the **retrieval encoder's own tokens**, not characters:
an embedding model truncates at a fixed token count (128 for the default
multilingual MiniLM), and Russian text costs more tokens per character than
English — so a character budget silently drops the tail of every chunk at index
time and behaves differently per language. `chunk_documents()` loads the encoder's
tokenizer automatically and falls back to characters when it is unavailable
(offline). Splits happen on sentence boundaries, never mid-word.

Small encoder windows would starve the LLM of context, so retrieval is
**small-to-big**: the encoder-sized chunk is what gets embedded and matched, and
every piece of a split section or table carries its full parent passage in
`metadata["context"]` — that passage (deduplicated across chunks) is what the QA
prompt actually contains. Precise search and complete answers at the same time.

Each chunk records the section it belongs to, all sections it covers, page
numbers, block ids and any VLM description for filtering and tracing.

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
