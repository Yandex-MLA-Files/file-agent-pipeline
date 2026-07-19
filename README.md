# File Agent Pipeline

A Python RAG application for answering questions about user-provided files. It parses documents, splits extracted text into chunks, retrieves relevant context with BM25 and semantic search, and generates source-grounded answers through an OpenAI-compatible LLM endpoint.

Supported formats: Markdown, plain text, PDF, DOCX, HTML, XLSX, and PPTX.

## Quick start

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

Copy `.env.example` to `.env` and configure either Yandex AI Studio or a local OpenAI-compatible endpoint before generating answers. See [local inference setup](docs/local_inference.md) for local vLLM and SGLang examples.

## Hugging Face QA datasets

The dataset input layer expects the columns `id`, `question`, `answer`, and
`doc_ids`. Source files are downloaded from the dataset repository using the
exact repository-relative paths stored in `doc_ids`.

Run a one-row smoke test before starting the full generation:

```bash
uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --config-name default \
  --split train \
  --revision e6db4819d8a100328378d5085fc5817688c0d663 \
  --cache-dir /mnt/storage-1/file-agent-pipe/huggingface \
  --output-dir /mnt/storage-1/file-agent-pipe/runs/smoke-001 \
  --limit 1 \
  --resume
```

For a full run, remove `--limit 1` and use a new output directory such as
`/mnt/storage-1/file-agent-pipe/runs/full-001`. The `--resume` flag is safe on
the first run and reuses matching row checkpoints after an interruption.

The command reads LLM settings from `.env`. It writes per-row checkpoints,
`answers.parquet`, the reloadable `hf_dataset/` directory, and
`run_manifest.json`. The manifest records the dataset selection, row-content
hash, model and retrieval parameters, prompt hash, and processed/resumed
counts; it never stores API keys or Hugging Face tokens.

The same pipeline can also be called from Python:

```python
from file_agent.hf_batch import generate_hf_qa_records
from file_agent.hf_dataset import load_qa_dataset
from file_agent.hf_output import save_generated_qa_dataset
from file_agent.llm.factory import create_llm_client

dataset = load_qa_dataset(
    dataset_id="sandrik1271/RAG-QA-Dataset",
    config_name="default",
    split="train",
    revision="e6db4819d8a100328378d5085fc5817688c0d663",
)
batch_result = generate_hf_qa_records(
    dataset=dataset,
    dataset_id="sandrik1271/RAG-QA-Dataset",
    revision="e6db4819d8a100328378d5085fc5817688c0d663",
    llm_client=create_llm_client(),
    output_dir="runs/baseline-001",
    resume=True,
)
artifacts = save_generated_qa_dataset(
    source_dataset=dataset,
    records=batch_result.records,
    output_dir="runs/baseline-001",
)
```

Each completed row is written atomically to the run's `checkpoints/`
directory. Restarting with `resume=True` validates and reuses matching
checkpoints, then continues from the first missing row.

After the batch is complete, `save_generated_qa_dataset` validates every
generated row against the source dataset and writes two final artifacts:
`answers.parquet` for tabular tools and `hf_dataset/` for loading with
`datasets.load_from_disk()`. Existing final artifacts are never overwritten.

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
