# Document agent

The agent answers questions about uploaded documents with a multi-step
think-act-observe loop instead of a single fixed retrieval. The LLM decides
which tool to call, reads the observation, and iterates until it can answer
(or the step budget runs out).

```text
question
  -> LLM plans a step
  -> tool call (search / overview / read a section)
  -> observation appended to the conversation
  -> ... repeat up to max_steps ...
  -> final answer + collected sources
```

## Components

- `src/file_agent/agent/tools.py` — the toolset over an indexed document
  collection:
  - `search_documents(query, top_k)` — hybrid retrieval via the shared
    `Retriever` interface; returns deduplicated parent passages with source
    file, section and pages.
  - `list_documents()` — file names, page counts and tables of contents.
  - `read_section(file_name, section)` — a whole section by heading, for
    summarization questions that retrieval handles poorly.
- `src/file_agent/agent/agent.py` — `FileAgent`, the orchestration loop, and
  `answer_with_agent(...)`, the convenience entry point used by the app.

Tools validate their arguments and raise `ToolError` with a message written
for the model, so the agent can correct itself on the next step. Unknown
tools, malformed replies and tool crashes are also fed back as observations
instead of aborting the run.

## Tool-call protocol

Tool calls are parsed on the client side from the model's reply:

```text
Thought: <one sentence>
Action: {"tool": "search_documents", "arguments": {"query": "..."}}
```

or

```text
Thought: <one sentence>
Final Answer: <answer in the language of the question>
```

Bare JSON tool calls without the `Action:` marker are accepted too, and a
reply with neither marker is treated as the final answer, so smaller models
degrade gracefully instead of looping on format reminders.

Client-side parsing is a deliberate choice: it works with any
OpenAI-compatible backend and does not depend on server-side tool-call
parsing. In particular, vLLM 0.7.3 (the newest version that still runs on
V100 GPUs — later versions pull in PyTorch builds without V100 support)
cannot combine `--enable-auto-tool-choice` with `--enable-reasoning`, so
native function calling plus reasoning is unavailable there. The agent needs
neither flag: reasoning output (`<think>...</think>` blocks) is stripped
before parsing, and the JSON action is extracted from the visible text.

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

## Limits and behavior

- `max_steps` (default 6) bounds the number of LLM turns; when the budget is
  exhausted the agent demands a final answer from the observations gathered
  so far.
- Observations are size-bounded at every level (`MAX_SECTION_CHARS`, capped
  `top_k`, and a hard `MAX_OBSERVATION_CHARS` cut in the loop), so the
  conversation stays inside the model's context window; the UI still sees
  the full tool output via the step record.
- Sources from every `search_documents` call are collected and deduplicated
  by chunk id; the UI shows them like the single-pass RAG sources.
- Every run is traced with OpenTelemetry: `file_agent.agent_run` wraps the
  loop, `file_agent.agent_tool` wraps each tool call, and the usual
  `file_agent.llm_generate` spans cover each model turn.

## Running

In the Streamlit app, choose the "Agent (multi-step)" answer mode. The LLM
backend is the regular one from `.env` (`LLM_BACKEND`, see
`docs/local_inference.md` for local serving).

Programmatic use:

```python
from file_agent.agent import answer_with_agent
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_llm_client
from file_agent.rag import index_documents, load_documents

documents = load_documents(["report.pdf"])
retriever = LanceDBRetriever()
index_documents(documents, retriever)

response = answer_with_agent(
    question="О чем раздел с результатами?",
    llm_client=create_llm_client(),
    retriever=retriever,
    documents=documents,
)
print(response.answer)
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
- The agent adds LLM turns (typically 2-4 per question), not indexing
  work: each question costs a few model calls against the already-built
  index.

Why OCR into the index instead of sending pages straight to a multimodal
LLM: parsing happens once per document, while questions are many — an
index amortizes the OCR cost, keeps question latency independent of
document size, and gives every answer a citable source (file, section,
page). Feeding page images to a VLM at question time re-reads the document
on every question, scales cost with page count, and cannot be searched.
The bounded VLM enhancer (`VLM_BACKEND`) remains the right place for
image-only content: it turns figures into indexed, searchable text — the
same amortized model.

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
