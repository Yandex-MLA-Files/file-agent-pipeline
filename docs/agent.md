# Document agent

The agent answers questions about uploaded documents with a multi-step
think-act-observe loop instead of a single fixed retrieval. The LLM decides
which tool to call, reads the observation, and iterates until it can answer
(or the step budget runs out). Everything a tool shows the model is a
labelled passage; the final answer names the passages it relies on, and those
become the answer's sources.

```text
question
  -> LLM plans a step
  -> one or several tool calls (search / exact text / overview / read / compute / look at a figure)
  -> observation: passages [P1], [P2], ... appended to the conversation
  -> ... repeat up to max_steps ...
  -> Final Answer + Sources: P2, P5
  -> (optional) editor pass over the draft against the cited passages
  -> answer + cited passages as sources
```

## Components

- `src/file_agent/agent/agent.py` — `FileAgent`, the orchestration loop;
  `AgentSettings` (environment-configurable knobs); `answer_with_agent(...)`,
  the entry point used by the app and by `--answer-mode agent`, which wraps
  the loop in a fallback to single-pass RAG.
- `src/file_agent/agent/tools.py` — the toolset over an indexed document
  collection (below).
- `src/file_agent/agent/passages.py` — `PassageRegistry`: one id space per
  run for everything the model has read; citation resolution.
- `src/file_agent/agent/tables.py` — DataFrames out of spreadsheet sheets and
  Markdown tables inside parsed documents (two-row headers merged, numeric
  columns converted).
- `src/file_agent/agent/sandbox.py` — guarded Python execution for
  `query_table` / `calculate`: no imports, no private attributes, no pandas /
  numpy I/O entry points, a builtin whitelist, a wall-clock limit and an output
  cap. A guard against careless model code, not a security boundary — the
  data it touches is the user's own upload, already in memory.

This branch runs the agent on the retrieval baseline (`feat/retrieval`):
structured parsers, VLM page OCR and formula enrichment, section-token
small-to-big chunking, lemmatised BM25 + dense hybrid with HyDE multi-query,
a cross-encoder reranker and distinct parent passages — all inside
`retriever.search()`. The agent's own alternative formulations fuse *on top*
of that stack (each formulation goes through the full pipeline, the lists
meet by reciprocal rank), and a per-document search rides the retriever's
native `source_file` restriction instead of over-fetching and filtering.

Tools validate their arguments and raise `ToolError` with a message written
for the model, so the agent can correct itself on the next step. Unknown
tools, malformed replies and tool crashes are also fed back as observations
instead of aborting the run; an identical repeated call is refused with a
note, and the model is warned one call before its budget ends. When the budget
is exhausted an answer is demanded; a model that still replies with a tool
call gets that one call (it is usually the last piece of a computation) and
the demand is repeated, and a reply that is still not an answer yields no
answer text rather than the model's notes.

## Tools

| tool | what it does | when the model is told to use it |
|---|---|---|
| `search_documents(query, queries?, top_k?, file_name?)` | hybrid retrieval through the shared `Retriever` (which itself runs HyDE multi-query, reranking and passage dedup); the main query plus up to three alternative formulations are searched separately and fused by reciprocal rank (`1/(60+rank)`); `file_name` restricts the search inside the index | first step for almost every question; `queries` carries synonyms, the document's own wording, the other language |
| `find_text(pattern, file_name?, max_hits?)` | exact case-insensitive substring (ё/е-tolerant, flexible whitespace) or regular-expression search over every block; a hit in a short block (a DOCX paragraph, a list item) is shown with its neighbouring blocks, a hit in a spreadsheet returns the whole row under its header row | numbers, codes, identifiers, names, dates, rare terms, quoted phrases |
| `list_documents()` | per document: type, pages/slides/sheets, sheet sizes with their columns, number of stored images, table of contents with page numbers, a short preview | structure questions, "how many sections / sheets", not knowing where to look |
| `read_section(file_name, section, part?)` | a heading and its body up to the next same-or-higher heading, served in parts of 7 000 characters; headings match exactly, by substring, or by most of their words | summarising or enumerating a whole section |
| `read_pages(file_name, pages)` | full text of pages (PDF/DOCX), slides (PPTX) or sheets by number (XLSX), one passage per page | the neighbourhood of a passage, a page the question names |
| `read_document(file_name, part?)` | the whole document as Markdown, in parts | short documents, introduction/conclusion, last resort |
| `query_table(file_name, code?, sheet?, header_row?)` | pandas code over `df` (the selected sheet/table), `sheets` (this document's) and `files[...]` (every other document's tables, for joins); empty code returns shape, columns, dtypes and the first rows | totals, counts, averages, maxima, unique values, joins across files, exact row lookups in large tables |
| `calculate(expression)` | arithmetic / Python expression with `math` | ratios, percentages, differences |
| `inspect_image(file_name, image?, question?)` | asks the vision model (the serving multimodal LLM) about a figure whose pixels survived parsing; without `image` it lists a document's figures with their captions | charts, diagrams, photos - anything the text around a figure does not spell out; registered only when some figure kept its pixels |

Every tool registers what it showed as a passage, so an answer built from a
section, a grep hit or a computed total is grounded in exported contexts
exactly like one built from retrieved chunks.

## Tool-call protocol

Tool calls are parsed on the client side from the model's reply:

```text
Thought: <one sentence>
Action: {"tool": "search_documents", "arguments": {"query": "...", "queries": ["...", "..."]}}
```

Up to `AGENT_MAX_PARALLEL_ACTIONS` independent calls may be issued at once as
a JSON list (`Action: [{...}, {...}]`) — the usual case is one search per
document of a two-document question. The final reply is

```text
Thought: <one sentence>
Final Answer: <answer in the language of the question>
Sources: P2, P5
```

The `Sources` line is stripped from the answer and resolved against the
registry; the answer text itself must not carry file names, page or section
numbers or passage ids. Those were the single largest loss of the previous
agent version under an answer judge: 86 % of its answers named their source
("раздел 3.2, стр. 19–20"), which the judge treats as claims it cannot verify
against the contexts — faithfulness 0.739 on those answers against 0.824 on
the rest. With citations the location lives in structured data (the UI lists
the sources; the evaluation exports them as contexts) and the text stays pure.

Bare JSON tool calls, fenced ```json blocks, `Actions:` lists and replies
with neither marker (treated as the final answer) are accepted too, so
smaller models degrade gracefully instead of looping on format reminders.
Reasoning output (`<think>...</think>`) is stripped before parsing.

Client-side parsing is a deliberate choice: it works with any
OpenAI-compatible backend and does not depend on server-side tool-call
parsing (vLLM 0.7.3, the newest version that still runs on V100 GPUs, cannot
combine `--enable-auto-tool-choice` with `--enable-reasoning`).

## Sources and the editor pass

`AgentResponse.sources` holds the cited passages in citation order; when the
model cites nothing, everything it read (bounded to eight passages) is
exported instead; `AGENT_CITED_SOURCES_ONLY=false` always exports cited
passages first and the rest after them.

`AGENT_VERIFY=true` adds one model call after the draft: the editor sees the
question, the cited passages and the draft, and either replies `KEEP` or
returns a corrected answer (unsupported statements removed, specifics the
passages provide added, location remarks dropped). A reply much shorter than
a long draft is treated as a misfire and ignored; an error keeps the draft.

`AGENT_FALLBACK_TO_RAG=true` answers with the single-pass QA prompt over the
collected passages (or a plain retrieval) when the loop raises or ends
without an answer, so the caller always gets one. `AGENT_TRACE_DIR` appends
one JSON line per question (steps, tool calls, observations, citations,
timings) to `<dir>/agent_trace.jsonl` for audits.

## Sessions (follow-up questions)

`AgentSession` gives the agent bounded conversation memory: the last
`max_turns` question/answer pairs are replayed as plain chat turns before
the current question, so follow-ups like "а во втором квартале?" resolve
against the previous exchange. Tool calls and observations from earlier
runs are deliberately not replayed — they are stale working state; the
agent re-queries the documents instead, which keeps the context window
small and the answers grounded in fresh observations.

In the Streamlit app the dialog persists while the same files stay
uploaded (a "Reset dialog" button clears it); re-uploading files starts a
fresh session.

```python
from file_agent.agent import AgentSession, answer_with_agent

session = AgentSession()
answer_with_agent(question="Какая выручка в первом квартале?", session=session, ...)
answer_with_agent(question="А во втором?", session=session, ...)
```

## Settings

| variable | default | meaning |
|---|---|---|
| `AGENT_MAX_STEPS` | 8 | model turns before an answer is demanded |
| `AGENT_VERIFY` | true | editor pass over the draft |
| `AGENT_CITED_SOURCES_ONLY` | true | export only cited passages as sources |
| `AGENT_FALLBACK_TO_RAG` | true | single-pass QA when the loop fails |
| `AGENT_MAX_PARALLEL_ACTIONS` | 3 | tool calls accepted from one reply |
| `AGENT_MAX_OBSERVATION_CHARS` | 14000 | hard cap on one observation |
| `AGENT_TRACE_DIR` | unset | per-question JSONL trace |

The settings and the prompt version are part of the generation fingerprint
of `--answer-mode agent` runs, so a checkpoint from another configuration is
never resumed. The LLM is the regular one from `.env` (`LLM_BACKEND`,
`LLM_ENABLE_THINKING=false` for Qwen3.5 served with a reasoning parser, a
generous `LLM_TIMEOUT_SECONDS` — one step may carry 30–40k characters of
observations).

## Evaluation

RESULTS_PLACEHOLDER

For reference, the same agent design over the *old* ingestion baseline
(branch `feat/agent`, unchanged legacy parsers and chunking) scored, on the
same judge and the same 122 rows: single-pass RAG
0.601 / 0.296 / 0.541 / 0.615 / 0.624, agent
0.863 / 0.469 / 0.869 / 0.885 / 0.761
(faithfulness / answer_correctness / answer_relevancy / context_precision /
context_recall). The editor pass earned its keep there - re-judging the same
run's draft answers dropped faithfulness from 0.86 to 0.77 - so
`AGENT_VERIFY` stays on by default.

A measurement note that carries over: `context_precision`/`context_recall`
for the agent grade the passages it *cited*, not everything it read -
precision is high by construction and recall is understated when the agent
read but did not cite a relevant passage; the metrics comparable one-to-one
with single-pass RAG are the three answer-side columns.

## Why not MCP (yet)

The toolset is deliberately plain Python behind the `Tool` dataclass. MCP
solves tool *distribution* — sharing one tool server between many agents
and hosts. Here the tools are process-local, bound to the in-memory
retriever and parsed documents of the current upload, so an MCP server
would add a transport layer, a serialization boundary for `SearchResult`
objects, and a new dependency without changing any behavior. If the tools
ever need to be shared with external agents (IDE assistants, other
services), wrapping `build_default_tools()` in an MCP server is the
natural next step — the `Tool` contract (name, description, parameters,
run) maps one-to-one onto an MCP tool definition.

## Running

The Streamlit app is a chat over the uploaded documents: upload files in the
sidebar, pick the answer mode there ("Agent (multi-step)" by default), and
ask questions in the chat box — follow-ups keep the dialog. Every answer
stays in the conversation with its citations, an expandable account of the
agent's steps (tool calls, observations, per-step timings) and the cited
passages as source cards; the library above the chat shows each document's
structure, parsing details and a text preview. Finished dialogs are kept in
`logs/chats/` and listed under "History" in the sidebar — a stored chat can
be reopened after a page reload or an app restart and continued against the
currently uploaded documents. For dataset generation add `--answer-mode
agent` (see `docs/hf_dataset_generation.md`).

Programmatic use:

```python
from file_agent.agent import answer_with_agent
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_llm_client
from file_agent.rag import index_documents, load_documents

documents = load_documents(["report.pdf", "sales.xlsx"])
retriever = LanceDBRetriever()
index_documents(documents, retriever)

response = answer_with_agent(
    question="Какой регион принёс наибольшую выручку?",
    llm_client=create_llm_client(),
    retriever=retriever,
    documents=documents,
)
print(response.answer)
print(response.citations)           # ["P3"]
for step in response.steps:
    print(step.tool, step.arguments)
```

## Performance notes

Measured on a mixed set of 10 real documents (~360 pages: digital and
scanned PDFs, DOCX, large Markdown):

- The one-time cost is parsing. Docling layout/table models plus OCR
  dominate indexing time; chunking (≤3 s), embedding (≤1.5 s per document
  after the model is warm) and hybrid search (≤60 ms per query) are noise
  in comparison.
- Parsing is model inference, so it follows the accelerator. On CPU a
  26-page digital PDF took ~140 s and an 83-page deck with 29 scanned
  pages did not finish within 35 minutes; on one GPU the same files took
  53 s and 76 s. If parsing feels slow, first check that
  `torch.cuda.is_available()` is true in the app's environment — a CUDA
  build mismatch silently drops the whole pipeline to CPU.
- Table-heavy financial PDFs are the slowest digital case (~2.3 s/page):
  table structure recognition is per-table model inference. It buys
  correctly ordered Markdown tables, which is what lets the LLM answer
  numeric questions reliably.
- The agent adds LLM turns, not indexing work: each question costs a few
  model calls against the already-built index (see the evaluation section
  for the measured step counts and timings).

Why OCR into the index instead of sending pages straight to a multimodal
LLM: parsing happens once per document, while questions are many — an
index amortizes the OCR cost, keeps question latency independent of
document size, and gives every answer a citable source (file, section,
page). Feeding page images to a VLM at question time re-reads the document
on every question, scales cost with page count, and cannot be searched.
The bounded VLM enhancer (`VLM_BACKEND`) remains the right place for
image-only content: it turns figures into indexed, searchable text — the
same amortized model.

OCR engine quality, measured on a real scanned lecture slide (Russian text
with formulas), original plus degraded variants:

- EasyOCR (default) reads Cyrillic correctly (confidence 0.72-0.81,
  ~3 s/page on GPU) and tolerates moderate skew; beyond roughly ±7° the
  characters survive but the *reading order* starts to scramble. Half
  resolution costs almost nothing.
- RapidOCR transliterates Cyrillic into Latin lookalikes ("Критерий" →
  "KpNTepnn") — its confidence stays high, so the failure is silent. Keep
  it only for Latin/CJK documents, as already noted in AGENTS.md.
- A multimodal LLM (vision-enabled Qwen3.5-9B on an OpenAI-compatible
  endpoint) transcribes the same page with correct structure and formulas
  as LaTeX, and is unaffected even by -12° skew — at ~20 s/page, roughly
  7x slower than EasyOCR. For badly skewed or photographed documents it is
  the quality ceiling; the practical integration point is the existing
  `VLM_BACKEND=openai` hook rather than replacing the per-page OCR router.
  Note the reference deployment serves the model with
  `--language-model-only`, which disables image input — a separate
  vision-enabled instance is required.

## Serving notes (vLLM on V100)

The agent works with plain `vllm serve <model>` — no tool-choice or
reasoning-parser flags are required:

```bash
vllm serve <model-or-path> --host 0.0.0.0 --port 8000
```

If the model emits reasoning (`<think>` blocks), leave it as-is; the agent
strips it client-side. Configure the app as usual:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://<host>:8000/v1
LOCAL_LLM_MODEL=<model-or-path>
LOCAL_LLM_API_KEY=
```
