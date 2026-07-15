# eval-pipeline

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/deps-uv-blueviolet)
![ruff](https://img.shields.io/badge/lint-ruff-red)

LLM-as-judge evaluation for RAG answers.

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

Independent of any specific dataset or RAG pipeline. The contract is a
single function: `Judge.evaluate(dataset)`, where `dataset` is a table with
columns `question` (X), `answer` (y_hyp), `contexts` (whatever), `ground_truth`
(y_ref). Where those four columns came from is not this package's concern.

## Install

```
uv sync
```

## Run

```
uv run python verify_setup.py
uv run python scripts/run_eval.py --run run.parquet --out reports/v1
```

`LLMJudge` talks to any OpenAI-compatible API, configured via 3 environment
variables: `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (see
`.env.example`). Switching providers (an open-weight model host today, an
internal endpoint later) is a matter of changing these values — no code
changes. These are read directly from the environment; `.env` is not
auto-loaded, export the values yourself or source the file before running.

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
    llm_judge.py       LLMJudge implementation — see docstrings for prompts and metric formulas
  report.py            aggregate scores into a report
scripts/run_eval.py     CLI entrypoint
tests/                  pytest suite, no network calls
verify_setup.py          one-shot smoke test
```


