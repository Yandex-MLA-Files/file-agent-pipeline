# Merging `feat/parsing-chunking` into `main`, and what it costs the agent branches

Measured on 2026-08-18 with `git merge-tree` (a dry run: nothing was merged,
rebased or pushed). Every branch below is the state of `origin/*` at that
moment.

## 1. The branch itself merges cleanly

`feat/parsing-chunking` is **89 commits ahead of `main` and 0 behind**, so
`main` moves forward without a merge commit and without conflicts.

## 2. What each agent branch will face afterwards

| Branch | commits since main | conflicts with `main` **today** | conflicts after the merge | of them new |
|---|---|---|---|---|
| `feat/agent` | 10 | 0 | 6 | 6 |
| `feature/agentic` | 32 | 0 | 19 | 19 |
| `feature/AgenticLangGraph` | 20 | 0 | 20 | 20 |
| `feature/LangGraph` | 1 | **4** | 4 | **0** |

All conflicts are content conflicts — no renames, no delete/modify, nothing
that needs archaeology.

`feature/LangGraph` is the exception in both directions: it branched from an
older `main` (30 commits behind), already conflicts on `app.py`,
`src/file_agent/rag.py`, `pyproject.toml` and `uv.lock`, and this branch adds
nothing to that list.

## 3. The conflicts, by what it takes to resolve them

The line counts are `+added/-removed` since the merge base, ours against
theirs — they say who rewrote the file and who touched it in passing.

### 3.1 Ours is a rewrite; take ours, re-apply their edit on top

These files were rewritten by this branch (the structured parsers, the
chunker, the pipeline, the retriever), and the agent branches changed a few
lines in them. Resolving means keeping our version and re-applying the small
change — usually a parameter, a log line, or a call site.

| File | ours | `feature/agentic` | `feature/AgenticLangGraph` | `feat/agent` |
|---|---|---|---|---|
| `src/file_agent/chunking.py` | +822/−107 | +39/−63 | +121/−17 | — |
| `src/file_agent/pipeline.py` | +259/−28 | +18/−25 | — | — |
| `src/file_agent/parsers/pptx_parser.py` | +395/−53 | +33/−1 | +11/−2 | — |
| `src/file_agent/parsers/xlsx_parser.py` | +328/−30 | — | +2/−1 | — |
| `src/file_agent/parsers/md_parser.py` | +212/−15 | — | — | +70/−7 |
| `src/file_agent/parsers/enhancer.py` | +131/−41 | +15/−9 | — | — |
| `src/file_agent/lancedb_retriever.py` | +149/−5 | +49/−18 | +50/−6 | — |

`md_parser.py` is worth naming separately: `feat/agent` made the Markdown
parser heading-aware, and this branch did the same thing independently and
went further. Their change is contained in ours; the conflict is textual.

`lancedb_retriever.py` was the one conflict in this group with real content on
both sides: `feature/agentic` and `feature/AgenticLangGraph` each added a
`source_file` column and a `search(..., source_file=…)` filter, which an
agent's per-document tool cannot work without, while this branch rewrote
`search()` around reranking and diversification. Taking either side whole would
have lost the other. It is therefore **implemented here** (as a prefilter, with
diversification switched off for a single-document query), so both branches can
resolve that file by taking ours and deleting their own version — and any
langfuse span attributes or locking they added around it still have to be
re-applied on top.

### 3.2 Both sides added features to the same file; merge by hand

| File | ours | `feature/agentic` | `feature/AgenticLangGraph` | `feat/agent` |
|---|---|---|---|---|
| `src/file_agent/hf_batch.py` | +114/−19 | +277/−48 | +73/−18 | +5/−0 |
| `src/file_agent/hf_cli.py` | +27/−0 | +79/−3 | +97/−5 | +13/−0 |
| `src/file_agent/llm/openai_client.py` | +13/−0 | +171/−30 | +183/−9 | — |
| `src/file_agent/llm/factory.py` | +56/−2 | +20/−2 | +65/−6 | — |
| `src/file_agent/vlm/factory.py` | +45/−13 | +3/−13 | +52/−0 | — |
| `src/file_agent/vlm/openai_compatible.py` | +59/−9 | +9/−2 | +44/−8 | — |
| `eval_pipeline/eval/judge/ragas_judge.py` | +18/−0 | — | +278/−41 | — |

Ours in this group is small and additive (a timeout, a thinking flag, a
`chat()` method, an extra client option); theirs is the substance. Take
theirs and re-apply our lines.

### 3.3 Tests

`tests/test_pptx_parser.py`, `test_xlsx_parser.py`, `test_md_parser.py`,
`test_hf_batch.py`, `test_hf_cli.py`, `test_hf_rag.py`, `test_retrieval.py`.
Same rule as the module each of them covers: where we rewrote the module, our
tests replace theirs; where they added a feature, their tests are the ones
that matter.

### 3.4 Configuration and documentation — mechanical

`.env.example`, `eval_pipeline/.env.example`, `README.md`, `AGENTS.md`,
`eval_pipeline/README.md`, `pyproject.toml`: both sides appended their own
blocks, so the resolution is to keep both.

`uv.lock` should not be merged by hand at all — resolve `pyproject.toml`
first, then regenerate:

```bash
uv lock
```

## 4. Suggested order for a teammate

```bash
git fetch origin
git switch feature/<yours>
git rebase origin/main          # or: git merge origin/main
```

Rebase gives smaller, per-commit conflicts; merge gives one large one. For
this shape of change — one side rewrote the ingestion core, the other added an
agent on top — the per-commit conflicts are easier, because most of an agent
branch never touches ingestion at all: of `feature/agentic`'s 32 commits only
a handful reach these files, and `feature/AgenticLangGraph` keeps its agent in
`src/file_agent/agent_tools.py` and `rag_graph.py`, which this branch does not
have and therefore cannot conflict with.
