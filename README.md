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

```python
from file_agent.hf_dataset import (
    QADatasetRecord,
    load_qa_dataset,
)
from file_agent.hf_rag import process_hf_qa_record
from file_agent.llm.factory import create_llm_client

dataset = load_qa_dataset(
    dataset_id="sandrik1271/RAG-QA-Dataset",
    config_name="default",
    split="train",
    revision="e6db4819d8a100328378d5085fc5817688c0d663",
)
record = QADatasetRecord.from_row(dataset[0])
generated_record = process_hf_qa_record(
    record=record,
    dataset_id="sandrik1271/RAG-QA-Dataset",
    revision="e6db4819d8a100328378d5085fc5817688c0d663",
    llm_client=create_llm_client(),
)
output_row = generated_record.to_dict()
```

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
