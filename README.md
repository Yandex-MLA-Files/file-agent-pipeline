# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

PDF and DOCX parsing uses Docling. OCR is enabled automatically for scanned PDF
pages. Optional VLM processing is disabled by default.

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

## Generate a Hugging Face dataset

The input dataset must contain `id`, `question`, `answer`, and `doc_ids`.
Files listed in `doc_ids` are downloaded from the same dataset repository.

```bash
uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --output-dir runs/smoke-001 \
  --limit 1 \
  --resume
```

Remove `--limit 1` for a full run. Use a new output directory for each run.
`--resume` continues an interrupted run from saved checkpoints.

The output directory contains:

- `answers.parquet` - final dataset;
- `hf_dataset/` - dataset for `datasets.load_from_disk()`;
- `run_manifest.json` - run parameters;
- `checkpoints/` - intermediate results used by `--resume`.

Remote server instructions:
[docs/hf_dataset_generation.md](docs/hf_dataset_generation.md).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
