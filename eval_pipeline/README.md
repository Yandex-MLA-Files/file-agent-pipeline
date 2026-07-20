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

`RagasJudge` talks to any OpenAI-compatible API, configured via 3 environment
variables: `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (see
`.env.example`). Switching providers (an open-weight model host today, an
internal endpoint later) is a matter of changing these values — no code
changes. These are read directly from the environment; `.env` is not
auto-loaded, export the values yourself or source the file before running.

Every `evaluate()` call appends a cost line to a usage log (`usage_log.jsonl`
by default) with token counts and RUB cost. Configurable via
`JUDGE_USAGE_LOG_PATH` and `JUDGE_PRICE_PER_1K_{INPUT,OUTPUT,CACHED}_TOKENS`
(see `.env.example`, and the module docstring in `ragas_judge.py` for why
cached tokens are billed separately).

`--out` only ever holds the latest report for that path -- rerunning with
the same `--out` overwrites it. `run_eval.py` also appends a one-line
summary (timestamp, run file, out dir, judge, per-metric means) to
`runs_log.jsonl` (path configurable via `--runs-log`) on every run, so
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


