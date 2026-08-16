# Hugging Face dataset generation on the remote server

This guide runs the RAG pipeline with a local Qwen model served by vLLM.
Commands assume the repository is in `~/file-agent-pipeline` and shared storage
is mounted at `/mnt/storage-1/file-agent-pipe`.

## 1. Prepare the project

```bash
cd ~/file-agent-pipeline

git pull --ff-only

source "$HOME/.local/bin/env"

export FILE_AGENT_STORAGE=/mnt/storage-1/file-agent-pipe
export UV_CACHE_DIR="$FILE_AGENT_STORAGE/uv-cache/$USER-$(hostname -s)"
export HF_HOME="$FILE_AGENT_STORAGE/huggingface/pipeline"
export XDG_CACHE_HOME="$FILE_AGENT_STORAGE/cache"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$HF_HOME" \
  "$XDG_CACHE_HOME" \
  "$FILE_AGENT_STORAGE/huggingface/vllm" \
  "$FILE_AGENT_STORAGE/runs" \
  "$FILE_AGENT_STORAGE/logs"

uv sync --frozen
```

Configure `.env`:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_API_KEY=
LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

VLM_BACKEND=off
OCR_ENGINE=easyocr
OCR_LANGS=ru,en
```

## 2. Start vLLM

Check which GPU is free:

```bash
nvidia-smi
```

If the `file-agent-vllm` container already exists and its configured GPU is
free, start it:

```bash
docker start file-agent-vllm
```

For the first start, create the container. The example uses GPU `0`; replace it
with the GPU assigned for the run.

```bash
docker run -d \
  --name file-agent-vllm \
  --gpus '"device=0"' \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$FILE_AGENT_STORAGE/huggingface/vllm:/root/.cache/huggingface" \
  -e VLLM_USE_V1=0 \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  vllm/vllm-openai:v0.7.3 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Wait until the endpoint is ready:

```bash
docker logs --tail 100 -f file-agent-vllm
```

Stop log output with `Ctrl+C`, then check the endpoint:

```bash
curl -fsS http://127.0.0.1:8000/health \
  && echo "vLLM is ready"
```

The port is bound to `127.0.0.1` and does not need to be forwarded in VS Code.

## 3. Run generation

Use `tmux` so the run continues after the SSH connection closes:

```bash
tmux new -s hf-generation
```

Inside `tmux`, repeat the exports from step 1 and set the local backend:

```bash
cd ~/file-agent-pipeline
source "$HOME/.local/bin/env"

export FILE_AGENT_STORAGE=/mnt/storage-1/file-agent-pipe
export UV_CACHE_DIR="$FILE_AGENT_STORAGE/uv-cache/$USER-$(hostname -s)"
export HF_HOME="$FILE_AGENT_STORAGE/huggingface/pipeline"
export XDG_CACHE_HOME="$FILE_AGENT_STORAGE/cache"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

export LLM_BACKEND=local
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_LLM_API_KEY=
export LOCAL_LLM_MODEL=Qwen/Qwen3.5-27B
export LOCAL_LLM_ENABLE_THINKING=false
export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export VLM_BACKEND=off
export OCR_ENGINE=easyocr
export OCR_LANGS=ru,en
```

For `tool_agent`, the Qwen endpoint must expose native OpenAI-compatible tool
calls. Start Qwen 3.5 with automatic tool choice and its matching tool-call
parser, as described in [`local_inference.md`](local_inference.md).

Create a new run directory:

```bash
export RUN_NAME=pilot20-qwen35-27b-standard-001
export RUN_DIR="$FILE_AGENT_STORAGE/runs/$RUN_NAME"
export RUN_LOG="$FILE_AGENT_STORAGE/logs/$RUN_NAME.log"
```

Generate answers for the first 20 rows:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES="" uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --split train \
  --cache-dir "$HF_HOME" \
  --output-dir "$RUN_DIR" \
  --rag-mode standard \
  --limit 20 \
  --top-k 5 \
  --temperature 0 \
  --max-tokens 1200 \
  --resume \
  2>&1 | tee "$RUN_LOG"
```

Generate the same rows with the tool agent into a different run directory:

```bash
export RUN_NAME=pilot20-qwen35-27b-tool-agent-001
export RUN_DIR="$FILE_AGENT_STORAGE/runs/$RUN_NAME"
export RUN_LOG="$FILE_AGENT_STORAGE/logs/$RUN_NAME.log"

CUDA_VISIBLE_DEVICES="" uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --split train \
  --cache-dir "$HF_HOME" \
  --output-dir "$RUN_DIR" \
  --rag-mode tool_agent \
  --max-tool-rounds 8 \
  --limit 20 \
  --top-k 5 \
  --temperature 0 \
  --max-tokens 1200 \
  --resume \
  2>&1 | tee "$RUN_LOG"
```

Always use separate output directories for the two modes. The mode, tool-round
limit, model settings, prompt hash, and retrieval settings are included in every
checkpoint and in `run_manifest.json`; `--resume` rejects a changed configuration.
For the final comparison, keep the dataset revision, embedding model, VLM/OCR
settings, chunking, `top-k`, temperature, and completion limit identical between
the two modes. The larger tool-round limit applies only to `tool_agent`.
For Qwen3.5 evaluation runs, `1200` completion tokens is enough for the benchmark's
short answers and tool-call JSON while reducing the impact of a rare repetitive
completion. The tool agent also rewrites excessively long or repeated final answers.

For a full run, use a new `RUN_NAME` and remove `--limit 20`. Add
`--revision <dataset-commit>` when the run must use a fixed dataset snapshot.
For a targeted pilot, replace `--limit` with one or more exact IDs, for example
`--record-id q0052 --record-id q0115 --record-id q0122`.

Detach from `tmux` with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t hf-generation
```

To continue an interrupted run, repeat the same command with the same
`RUN_DIR` and `--resume`. The generation parameters must remain unchanged.

## 4. Output files

A completed run is stored in:

```text
/mnt/storage-1/file-agent-pipe/runs/<RUN_NAME>/
```

It contains:

- `answers.parquet` - the final table. Along with `id`, `question`, `doc_ids`,
  `answer_model`, `contexts`, and `answer`, it records `rag_mode`, `stop_reason`,
  `search_queries`, `tool_calls_json`, LLM call/token counters, and row duration;
- `hf_dataset/` - the same table saved for
  `datasets.load_from_disk(<path>)`;
- `run_manifest.json` - dataset information, model and generation parameters,
  row counts, and artifact names;
- `checkpoints/` - one JSON result per completed row, used by `--resume`.

`answer_model` is the generated answer and `answer` is the reference answer
from the source dataset. Each item in `contexts` contains:

- `text` - the passage sent to the LLM;
- `retrieval_text` - the smaller chunk matched by the retriever;
- `document_id`, `rank`, and `score` - source and retrieval information;
- `metadata_json` - page, block type, source coordinates, and other metadata.

For `tool_agent`, `contexts[].text` is the exact evidence exposed to the model:
the bounded search preview, the full passage after `read_source_context`, or the
content returned by a direct page/section/table/visual read. Batch generation
rejects an agent answer that did not call a document evidence tool.

The run log is stored separately in:

```text
/mnt/storage-1/file-agent-pipe/logs/<RUN_NAME>.log
```

Quick result check:

```bash
uv run python - <<'PY'
import os
from datasets import load_from_disk

dataset = load_from_disk(f'{os.environ["RUN_DIR"]}/hf_dataset')

print("Rows:", len(dataset))
print("Columns:", dataset.column_names)
print("Empty model answers:", sum(not row["answer_model"].strip() for row in dataset))
print("Rows without contexts:", sum(not row["contexts"] for row in dataset))
PY
```

After the run, release the GPU:

```bash
docker stop file-agent-vllm
```
