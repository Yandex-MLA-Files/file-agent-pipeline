# Agent instructions

## Setup

```
uv sync
```

## Test

```
uv run pytest tests/ -v
```

## Lint and format

```
uv run ruff check .
uv run ruff format .
```

Run both before committing. CI (once added) will fail on lint errors or
unformatted code.

## Conventions

- All code, comments, docstrings, and commit messages in English.
- No filler phrases, no decorative emoji, no restating what the code
  already makes obvious. Keep prose dry and specific.
- `README.md` covers what the project is, how to install it, how to run
  it — nothing else. Implementation details (prompts, formulas, exact
  schemas) live in code docstrings, not in the README. Don't duplicate one
  in the other.
- `eval/` must stay independent of any specific dataset or RAG pipeline.
  It only knows about a table shaped `(id, question, answer, contexts,
  ground_truth)`. Do not add imports, docs, or logic that assume a
  particular dataset's schema or a particular pipeline's internals.
- Prefer adding a test alongside any behavior change, including one that
  fails without the change.
