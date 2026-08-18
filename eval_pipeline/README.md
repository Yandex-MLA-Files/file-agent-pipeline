# eval-pipeline

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/deps-uv-blueviolet)
![ruff](https://img.shields.io/badge/lint-ruff-red)

LLM-as-judge evaluation for RAG answers.

The project requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

Independent of any specific dataset or RAG pipeline. The contract is a
single function: `RagasJudge.evaluate(dataset)`, where `dataset` is a table with
columns `question` (X), `answer_model` (y_hyp), `contexts` (whatever),
`answer` (y_ref). Where those four columns came from is not this package's
concern.

## Install

```
uv sync
```

## Run

```
uv run python scripts/run_eval.py --run run.parquet --out reports/v1 --resume
```

The input is evaluated in batches of five rows by default. After every
successful batch, `reports/v1/checkpoint.parquet` is replaced atomically. If
the process is interrupted, run the same command with `--resume`: rows already
present in the checkpoint are skipped. Use `--batch-size N` to change the
maximum number of rows that may need to be repeated after an interruption.

The checkpoint manifest binds saved scores to the input Parquet content, judge
model, embedding model, metric list, and judge execution policy. Resume fails
safely if any of these change. A legacy checkpoint created before the execution
policy was recorded is accepted once and upgraded in place, preserving all of
its completed rows and recording their count as `legacy_completed_rows`.
Running without `--resume` intentionally starts the output directory from
scratch and removes its old checkpoint and final report.

`RagasJudge` talks to any OpenAI-compatible API for the judge LLM, configured
via `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL` (see `.env.example`).
Switching providers (an open-weight model host today, an internal endpoint
later) is a matter of changing these values — no code changes. These are
read directly from the environment; `.env` is not auto-loaded, export the
values yourself, `source` the file, or run with
`uv run --env-file .env ...`.

Judge requests can be bounded with `JUDGE_MAX_TOKENS` and
`JUDGE_REQUEST_TIMEOUT`; `JUDGE_TIMEOUT` remains the timeout for a complete
Ragas metric job, which may contain more than one LLM request. For reasoning
models, leave `JUDGE_REASONING_MODE` empty to retain the provider default on
the primary attempt and set `JUDGE_FALLBACK_REASONING_MODE=DISABLED` to retry
only metrics that returned an empty or non-numeric score. Metrics that already
succeeded are retained and are not requested again. The next question starts
again with the primary reasoning mode. Yandex AI Studio's native SDK exposes
`reasoning_mode`, but its OpenAI-compatible endpoint rejects that field. When
that endpoint is detected, the fallback is therefore a bounded retry with the
provider default instead of an HTTP 400 request.

Fallback use is explicit in both logs: `usage_log.jsonl` records
`fallback_metric_attempts`, while each judge trace records `fallback_metrics`,
`fallback_failed_metrics`, and an `attempt` / `reasoning_mode` annotation on
individual prompt calls. A fallback that is also incomplete is never replaced
with a fabricated zero; the row remains incomplete and the resumable runner
records it in `incomplete_rows.json`, continues evaluating later rows, and
exits without a final report after all other pending rows have been checked.
Running the same command with `--resume` later retries only those incomplete
IDs; all rows with five finite scores remain checkpointed.

`answer_relevancy` additionally needs an embeddings model -- this runs
**locally** (`JUDGE_EMBEDDING_MODEL`, sentence-transformers, default
`intfloat/multilingual-e5-small`), not through the judge API, so it's free
and doesn't depend on the judge provider exposing an embeddings endpoint.
First use downloads the model from HuggingFace Hub.

The five per-row score columns are `faithfulness`, `answer_correctness`,
`answer_relevancy`, `context_precision`, and `context_recall`. The
`answer_correctness` score uses Ragas' standard `AnswerCorrectness` metric.
Retriever chunk dictionaries are rendered with their source file and page
numbers so citation claims can be checked rather than treated as unsupported.

Datasets containing questions whose correct answer is "the source does not
say" need a separate view: standard Ragas `answer_relevancy` deliberately
penalizes noncommittal answers, including correct abstentions. Set
`JUDGE_NEGATIVE_EXAMPLE_PATTERN` (or pass `--negative-example-pattern`) to add
`mean_on_answerable_examples` and `mean_on_negative_examples` for every metric
without changing the five raw standard scores.

Every `evaluate()` call writes under `logs/` (gitignored, created
automatically if missing):

- `<out>/usage_log.jsonl` -- appended, one cost line per completed batch with
  token counts and RUB cost. Sum the batch entries belonging to an evaluation
  to get its locally estimated total. Configurable via `JUDGE_USAGE_LOG_PATH` and
  `JUDGE_PRICE_PER_1K_{INPUT,OUTPUT,CACHED}_TOKENS` (see `.env.example`, and
  the module docstring in `ragas_judge.py` for why cached tokens are billed
  separately).
- `<out>/judge_trace_log_<timestamp>.jsonl` -- a fresh file per `evaluate()`
  call (not appended -- each run gets its own file, so it can be
  opened/grepped/deleted independently), one line per evaluated example:
  `question`, `answer_model`, `reference`, `contexts`, per-metric `verdict`,
  and the full `reasoning_trace` (every intermediate judge prompt/response).
  Field names mirror the run file's own schema, so a human reviewer
  checking judge verdicts against ground truth doesn't need to
  cross-reference the original run file. The CLI defaults both log paths to
  its `--out` directory; the base names remain configurable via
  `JUDGE_USAGE_LOG_PATH` and `JUDGE_TRACE_LOG_PATH`.

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
    ragas_judge.py     RagasJudge — see module docstring for metric choices
  report.py            aggregate scores into a report
scripts/run_eval.py     CLI entrypoint
tests/                  pytest suite, no network calls
```
