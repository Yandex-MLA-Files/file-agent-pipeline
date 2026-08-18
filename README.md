# File Agent Pipeline

Agentic question-answering over user-provided documents (PDF, DOCX, PPTX,
XLSX, Markdown, TXT, HTML), with a plain single-shot RAG baseline kept
alongside it for comparison.

```text
files -> parsing -> chunking -> LanceDB retrieval -> agent loop -> answer
```

## How it works

**Baseline RAG** (`generate_baseline_rag_dataset.py` / `src/file_agent/baseline_cli.py`,
and the "no tools" path in `rag.py`): one retrieval pass, one LLM call. No
retries, no re-retrieval, no way to read more of a document than the top-k
retrieved chunks.

**Agentic pipeline** (`src/file_agent/agent/`): a ReAct tool-calling loop
(`agent/loop.py`) - the model reasons and calls tools itself (native OpenAI
`tools=`, not prompt-embedded JSON) for up to `AGENT_MAX_ITERATIONS` turns,
instead of a single fixed retrieve-then-generate pass. Five tools
(`agent/tools.py`):

| Tool | What it's for |
|---|---|
| `search_documents` | hybrid semantic + full-text search over the indexed chunks (the baseline's only retrieval mechanism, available here to call more than once) |
| `read_page` | full text of a specific page/slide, with pagination, when a search hit is relevant but incomplete |
| `list_documents` | document structure (pages/slides/sheets, headings, which pages have an undescribed figure) to orient before searching |
| `run_python` | sandboxed, network-disabled code execution over the uploaded files, each document's full extracted text, and any Docling-parsed tables - exact computation a text search can't do |
| `describe_image` | a vision-language model describes a figure/chart against the specific question asked (only registered when `VLM_BACKEND` is configured) |

Both the baseline and the agentic pipeline share the same parsing, chunking,
and LanceDB retrieval layers - only what happens after retrieval differs.
See [AGENTS.md](AGENTS.md) for the full architecture, module layout, and
design rules.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Configure the LLM backend in `.env` (`LLM_BACKEND=yandex` or `local`, see
comments in `.env.example` for every variable). For the `run_python` tool,
build the sandbox image once:

```bash
docker build -t file-agent-sandbox -f docker/sandbox.Dockerfile .
```

Without it, `run_python` degrades to a tool-observation error ("Docker CLI
not found") instead of crashing the agent - useful for a quick local check,
but table/spreadsheet questions won't get answered correctly without it.

## Run the chat UI

```bash
uv run streamlit run app.py
```

Upload files in the sidebar, then ask questions. Each chat keeps its own
documents and its own in-memory LanceDB index; "Generate answer" runs the
same `run_react_agent` as the batch pipeline below, not a separate
simplified path.

## Generate an evaluation dataset

Batch-run either pipeline over a Hugging Face QA dataset and write an
`answers.parquet` scoreable by `eval_pipeline` (see below):

```bash
# agentic pipeline
uv run python generate_hf_dataset.py --dataset-id <hf-dataset-id> --split train --output-dir runs/<run-name>

# baseline, for comparison - same output shape, no tools
uv run python generate_baseline_rag_dataset.py --dataset-id <hf-dataset-id> --split train --output-dir runs/<run-name>
```

Full flags, resuming an interrupted run, output file layout, and the
recommended remote-server workflow (vLLM setup, `tmux`, GPU selection):
[docs/hf_dataset_generation.md](docs/hf_dataset_generation.md). Local
inference endpoint setup: [docs/local_inference.md](docs/local_inference.md).

A second, independent evaluation set (DocBench, an external benchmark
labeled by question type) is documented separately:
[docs/docbench_evaluation.md](docs/docbench_evaluation.md).

## Evaluate

Scoring is a separate project with its own venv, `eval_pipeline/` (see its
own `README.md`) - an LLM-as-a-Judge harness (Ragas, five metrics:
faithfulness, answer correctness, answer relevancy, context precision,
context recall) over any `answers.parquet` produced above:

```bash
cd eval_pipeline
uv run --env-file .env python scripts/run_eval.py --run ../runs/<run-name>/answers.parquet --out reports/<run-name>
```

This calls a real, paid judge model - check `eval_pipeline/logs/usage_log.jsonl`
for actual historical costs before scoring a large run. Not every metric the
judge reports should be trusted equally; a human-agreement audit
(`eval_pipeline/tests/METRIC_RELIABILITY.md`, if present locally) found
faithfulness, answer relevancy, and context recall reliable, and
context precision / answer correctness not reliable enough to draw
conclusions from.

## Tracing

Two independent tracing systems, deliberately not merged:

**OpenTelemetry + Jaeger** - the parsing/chunking/retrieval layers
(`src/file_agent/telemetry.py`). Spans in `parse_file`, `chunk_document`,
`load_documents`, `index_documents`, `LanceDBRetriever.index`/`.search`, and
the plain-RAG answer path, exported over OTLP/gRPC.

```bash
docker compose up -d jaeger
```

UI at http://localhost:16686 (service `file-agent-pipeline`). Spans go to
`localhost:4317` by default (`OTEL_EXPORTER_OTLP_ENDPOINT` to override). If
Jaeger isn't running, `configure_telemetry()` still works - spans are sent in
the background and simply don't arrive anywhere.

**Langfuse** - the agent loop specifically: one trace per question
(`agent/observability.py`'s `pipeline_trace`), one generation per LLM turn,
one span per tool call. Self-hosted, own stack:

```bash
docker compose up -d file-agent-langfuse-web file-agent-langfuse-worker
```

UI at http://localhost:3050 (get `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
from it after first bring-up, put them in `.env`). Left unset, agent tracing
is a silent no-op (e.g. in tests) - the agent works identically either way.

Both `generate_hf_dataset.py`/`generate_baseline_rag_dataset.py` and `app.py`
call `configure_telemetry()` before Langfuse's client initializes, so
Langfuse reuses the same global `TracerProvider` instead of creating its
own - every parsing/chunking/retrieval span above lands nested inside the
matching Langfuse trace automatically, no extra instrumentation needed.

`eval_pipeline`'s own run comparisons additionally log to MLflow when
`MLFLOW_TRACKING_URI` is set:

```bash
docker compose up -d file-agent-mlflow-eval
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

See [AGENTS.md](AGENTS.md) for architecture rules, module layout, and
contribution guidelines; [CONTRIBUTING.md](CONTRIBUTING.md) for the
contribution process itself.
