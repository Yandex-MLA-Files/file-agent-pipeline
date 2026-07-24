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
  born-digital pages stay fast and only scanned/image pages are OCR'd. OCR runs
  through RapidOCR (ONNX models bundled in the wheel, so it works offline).
  Override with `parse_file(path, enable_ocr="on" | "off")`.
- **Figures can be described by a VLM.** With `parse_file(path, enable_vlm=True)`,
  figures and diagrams are cropped and sent to an OpenAI-compatible vision model;
  the description is folded into the searchable text. VLM is off by default and
  degrades gracefully when no endpoint is reachable.

If Docling cannot process a PDF, the pipeline falls back to a plain PyMuPDF text
extraction so parsing never hard-fails.

## Chunking

Structured parsing yields many small blocks, so chunking **packs consecutive
blocks up to a size budget** instead of emitting one chunk per block — otherwise
retrieval returns a handful of tiny fragments with almost no context. Headings
stay with their section text, tables are kept whole, oversized blocks are split
into overlapping windows, and each chunk carries page numbers, block ids, the
section title and any VLM description for filtering and tracing.

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
