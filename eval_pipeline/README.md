# eval-pipeline

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/deps-uv-blueviolet)
![ruff](https://img.shields.io/badge/lint-ruff-red)

LLM-as-judge evaluation for RAG answers.

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

Independent of any specific dataset or RAG pipeline. The contract is a
single function: `Judge.evaluate(dataset)`, where `dataset` is a table with
columns `question` (X), `answer_model` (y_hyp), `contexts` (whatever),
`answer` (y_ref). Where those four columns came from is not this package's
concern.

## Install

```
uv sync
```

## Run

```
uv run python verify_setup.py
uv run python scripts/run_eval.py --run run.parquet --out reports/v1
```

`RagasJudge` talks to any OpenAI-compatible API for the judge LLM, configured
via `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (see `.env.example`).
Switching providers (an open-weight model host today, an internal endpoint
later) is a matter of changing these values — no code changes. These are
read directly from the environment; `.env` is not auto-loaded, export the
values yourself, `source` the file, or run with
`uv run --env-file .env ...`.

`answer_relevancy` additionally needs an embeddings model -- this runs
**locally** (`JUDGE_EMBEDDING_MODEL`, sentence-transformers, default
`intfloat/multilingual-e5-small`), not through the judge API, so it's free
and doesn't depend on the judge provider exposing an embeddings endpoint.
First use downloads the model from HuggingFace Hub.

Every `evaluate()` call writes under `logs/` (gitignored, created
automatically if missing):

- `logs/usage_log.jsonl` -- appended, one cost line per run with token
  counts and RUB cost. Configurable via `JUDGE_USAGE_LOG_PATH` and
  `JUDGE_PRICE_PER_1K_{INPUT,OUTPUT,CACHED}_TOKENS` (see `.env.example`, and
  the module docstring in `ragas_judge.py` for why cached tokens are billed
  separately).
- `logs/judge_trace_log_<timestamp>.jsonl` -- a fresh file per `evaluate()`
  call (not appended -- each run gets its own file, so it can be
  opened/grepped/deleted independently), one line per evaluated example:
  `question`, `answer_model`, `reference`, `contexts`, per-metric `verdict`,
  and the full `reasoning_trace` (every intermediate judge prompt/response).
  Field names mirror the run file's own schema, so a human reviewer
  checking judge verdicts against ground truth doesn't need to
  cross-reference the original run file. Base name configurable via
  `JUDGE_TRACE_LOG_PATH`.

`--out` only ever holds the latest report for that path -- rerunning with
the same `--out` overwrites it. `run_eval.py` also appends a one-line
summary (timestamp, run file, out dir, judge, per-metric means) to
`logs/runs_log.jsonl` (path configurable via `--runs-log`) on every run, so
separate runs -- e.g. different RAG versions -- can be compared without
having to remember a unique `--out` each time.

## Test

```
uv run pytest tests/ -v
```

## Lint

```
uv run ruff check .
uv run ruff format .
```

## Layout

```
eval/
  run_loader.py       load + validate a run file
  judge/
    base.py           Judge interface
    ragas_judge.py     RagasJudge implementation — see module docstring for metric choices
  report.py            aggregate scores into a report
scripts/run_eval.py     CLI entrypoint
tests/                  pytest suite, no network calls
verify_setup.py          one-shot smoke test
```


