# DocBench evaluation

Runs the ReAct agent over [DocBench](https://github.com/Anni-Zou/DocBench), an
external document-QA benchmark (229 PDFs, ~1074 questions, labeled by question
type: `text-only`, `meta-data`, `multimodal-t` (tables), `multimodal-f`
(figures), `unanswerable`, `una-web`). Used as a second, independent
evaluation set on top of the project's own `sandrik1271/RAG-QA-Dataset` runs
(see [hf_dataset_generation.md](hf_dataset_generation.md)) - a different
document corpus, language, and domain, to check whether results hold up
outside the main dataset. Scoring stays on this project's own RagasJudge
(DeepSeek-V4-Flash), not DocBench's own GPT-4 grader.

## 1. Get the data

Download from the link in the [DocBench repo](https://github.com/Anni-Zou/DocBench)
(Google Drive) and unpack so each folder sits directly under `data/`:

```text
data/
  0/
    0_qa.jsonl
    <some-paper>.pdf
  1/
    1_qa.jsonl
    <some-paper>.pdf
  ...
```

`data/` is entirely gitignored - nothing here is ever committed. The script
auto-discovers every numeric folder under `data/`, so partial downloads (only
some folders present) work fine, it just processes fewer documents.

## 2. Prerequisites

- A configured `file-agent-pipeline/.env` (same one `app.py`/`generate_hf_dataset.py`
  use - `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, etc.).
- A reachable LLM endpoint, e.g. the SSH tunnel to the shared cluster:
  `ssh -L 8000:10.128.0.40:8000 mla-yac-a100-5`. Keep it open for the whole run.
- `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env` are optional - if set,
  every question's trace is grouped under a Langfuse session named
  `docbench-pilot-002` (see `agent/observability.py`'s `pipeline_trace`).

## 3. Run

```bash
uv run python scripts/docbench_pilot.py
```

Questions within a document run concurrently (`MAX_CONCURRENT_QUESTIONS = 8`
in the script - tune down if the shared LLM endpoint is under load from
others). Safe to `Ctrl+C` and rerun the exact same command at any time:
already-answered questions are loaded from
`runs/docbench-pilot-002/checkpoints/` instead of recomputed. This is
deliberately simpler than `hf_batch.py`'s checkpointing (no schema
versioning or parameter-fingerprint matching) since this script isn't a
shared production artifact - delete the checkpoints directory if the
script's own code changes and you want a clean rerun.

If a question fails (LLM error, timeout after `QUESTION_TIMEOUT_SECONDS = 4
minutes`) it's checkpointed as `answer_model = "ERROR: ..."` and the run
continues - it does **not** retry automatically on the next run, since a
checkpoint already exists for it. To force a retry (e.g. after a dropped
tunnel caused a burst of connection errors), delete just the affected
`runs/docbench-pilot-002/checkpoints/<id>.json` files, not the whole
directory - a small one-off script over the checkpoint JSONs
(`answer_model.startswith("ERROR")`) is enough to find them.

## 4. Output files

```text
runs/docbench-pilot-002/
  answers.parquet     # id, question, answer_model, contexts, answer, type, folder
  checkpoints/        # one JSON per question, used for resume
```

`answers.parquet` is written after every folder completes, not only at the
end, so an interrupted run still leaves everything completed so far on disk.
`type` and `folder` are extra columns beyond `eval_pipeline`'s required
schema - carried through untouched by `RagasJudge.evaluate()`, useful for a
per-question-type breakdown afterward (`scored_df.groupby("type")[...]`).

## 5. Score it

From the `eval_pipeline` venv (a separate project/venv - see its own
`README.md`):

```bash
cd eval_pipeline
uv run --env-file .env python scripts/run_eval.py \
    --run ../runs/docbench-pilot-002/answers.parquet \
    --out reports/docbench-pilot-002
```

`--env-file .env` is required: `run_eval.py`/`ragas_judge.py` read
`JUDGE_MODEL`/`JUDGE_BASE_URL`/`JUDGE_API_KEY` from the environment but don't
load `.env` themselves.

This calls a real, paid judge API (DeepSeek-V4-Flash via Yandex Cloud) -
historically around 6-10₽ per row (see `eval_pipeline/logs/usage_log.jsonl`
for actual past costs). Check `answers.parquet`'s error rate first
(`answer_model.str.startswith("ERROR")`) before scoring a large run - there's
no reason to pay to judge rows that failed outright.

Output: `eval_pipeline/reports/docbench-pilot-002/report.json` (aggregate
means) and `scored.parquet` (per-row scores, still carrying `type`/`folder`
for breakdown). A one-line summary is also appended to
`eval_pipeline/logs/runs_log.jsonl`, alongside every other run ever scored in
this project - useful for comparing across runs over time.
