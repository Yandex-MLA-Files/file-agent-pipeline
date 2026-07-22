# File Agent Pipeline

RAG pipeline for answering questions about files. It parses documents, splits
them into chunks, retrieves relevant text with LanceDB, and sends the retrieved
context to an OpenAI-compatible LLM.

Supported formats: Markdown, TXT, PDF, DOCX, HTML, XLSX, and PPTX.

## Setup

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Configure the LLM in `.env`, then start the Streamlit application:

```bash
uv run streamlit run app.py
```

See [docs/local_inference.md](docs/local_inference.md) for local vLLM and
SGLang configuration.

## Generate answers for a Hugging Face dataset

The input dataset must contain `id`, `question`, `answer`, and `doc_ids`.
Files listed in `doc_ids` are downloaded from the same dataset repository.

Start with one row:

```bash
uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --output-dir runs/smoke-001 \
  --limit 1 \
  --resume
```

For a full run, remove `--limit 1` and use a new output directory. Keep
`--resume` to reuse completed rows after an interrupted run. Use `--revision`
to pin a dataset snapshot. Run with `--help` to see all options.

The output directory contains:

- `answers.parquet` — generated answers and retrieved contexts;
- `hf_dataset/` — the same result for `datasets.load_from_disk()`;
- `run_manifest.json` — dataset revision and generation parameters;
- `checkpoints/` — per-row results used by `--resume`.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
