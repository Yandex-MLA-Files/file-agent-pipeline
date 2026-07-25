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
- **Figures can be described by a VLM.** With `parse_file(path, enable_vlm=True)`,
  figures and diagrams are cropped and sent to an OpenAI-compatible vision model;
  the description is folded into the searchable text. VLM is off by default and
  degrades gracefully when no endpoint is reachable.

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
