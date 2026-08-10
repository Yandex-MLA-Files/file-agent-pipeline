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

## Limits and behavior

- `max_steps` (default 6) bounds the number of LLM turns; when the budget is
  exhausted the agent demands a final answer from the observations gathered
  so far.
- Observations are size-bounded (`MAX_SECTION_CHARS`, capped `top_k`), so
  the conversation stays inside the model's context window.
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
