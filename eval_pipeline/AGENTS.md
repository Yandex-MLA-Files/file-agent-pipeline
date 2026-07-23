# Project

Python project for an ML internship. Pipeline: user uploads a file and asks
a question, the system extracts the file's content and answers using an LLM.

Formats planned: PDF, PPTX, XLSX, Markdown, HTML.

## Current MVP

Already implemented:
- Markdown, PDF, HTML parsing
- unified Document / Block representation
- chunking
- keyword retrieval over chunks
- Streamlit interface: file upload, extracted-text preview, chunk search
- LLM QA prompt layer through an abstract llm_client

No access yet to Yandex Cloud / YandexGPT API. Keep developing without real
API calls until access arrives.

## Allowed

- build YandexGPTClient with no real requests in tests
- mock tests for YandexGPTClient
- FakeLLM mode to exercise the full pipeline without the external API
- new parsers (XLSXParser, PPTXParser)
- Streamlit UI improvements
- README.md improvements
- more tests
- better error handling

## Not allowed without discussion

- real requests to YandexGPT in pytest
- hardcoded API keys, folder_id, model_uri
- committing .env (.env.example is fine)
- OCR, VLM
- embeddings / FAISS
- complex agentic architecture

LangChain and LangGraph are in the dependency tree as of `ragas_judge.py`
(`eval_pipeline/eval/judge/ragas_judge.py`) — RAGAS pulls both in
transitively. They're not used directly anywhere in this project's own code;
this exception covers only what RAGAS itself requires.

Embeddings: discussed and approved for one narrow use -- `RagasJudge`'s
`answer_relevancy` metric (`ResponseRelevancy`) needs an embeddings model
(`JUDGE_EMBEDDING_MODEL`) to compare judge-generated questions against the
real one. Runs locally via `sentence-transformers` (new project dependency,
pulls in `torch`/`transformers`), not through the judge LLM's API -- no
external embeddings call, no extra cost. This does NOT extend to the main
RAG pipeline (parsing/chunking/retrieval) -- adding embeddings/FAISS there
is still a separate decision
that needs its own discussion.

Environment variables: `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_MODEL`.

## Stack

- Python 3.11+
- Streamlit
- PyMuPDF (PDF)
- BeautifulSoup (HTML)
- markdown (Markdown)
- pytest

## Architecture rules

- Parsing logic stays separate from LLM logic.
- All files normalize to the Document representation.
- Preserve metadata when available: `page` (PDF), `slide` (PPTX),
  `sheet` (XLSX), `block_type`.
- Don't add architecture ahead of need.
- Simple, readable code, type hints everywhere.
- Each parser gets tests.

## Commands

```
uv sync                    # setup
uv run pytest tests/ -v    # test
uv run ruff check .        # lint
uv run ruff format .       # format
```

Run lint and format before every commit. CI (once added) fails on lint
errors or unformatted code.

## Conventions

- All code, comments, docstrings, commit messages in English.
- No filler phrases, no decorative emoji, no restating what the code
  already makes obvious. Dry, specific prose.
- No AI-slop patterns: no "This function is responsible for handling...",
  no restating the diff in the commit message body, no comments that
  describe what the next line does when the line is already clear, no
  padding a docstring with a sentence that repeats the function name.
  If a comment doesn't add information a reader doesn't already have,
  delete it.
- `README.md` covers what the project is, how to install it, how to run
  it — nothing else. Implementation details (prompts, formulas, exact
  schemas) live in code docstrings, not in the README. Don't duplicate
  one in the other.
- `eval/` stays independent of any specific dataset or RAG pipeline. It
  only knows about a table shaped `(id, question, answer_model, contexts,
  answer)`. No imports, docs, or logic that assume a particular
  dataset's schema or a particular pipeline's internals.
- Prefer adding a test alongside any behavior change, including one that
  fails without the change.