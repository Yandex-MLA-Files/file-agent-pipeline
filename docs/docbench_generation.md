# DocBench generation

`generate_docbench.py` runs the file agent over a local checkout of
[Anni-Zou/DocBench](https://github.com/Anni-Zou/DocBench). It expects the official
directory layout:

```text
DocBench/data/
  0/
    <document>.pdf
    0_qa.jsonl
  1/
    <document>.pdf
    1_qa.jsonl
  ...
```

The official release contains 229 documents and 1,102 questions. The runner
discovers them locally and never sends the reference answer, evidence, question
type, or domain to the file agent. Those fields are copied only to the output so
the generated answer can be evaluated later.

## What the runner does

- parses, chunks, and indexes each PDF once, then reuses that in-memory index for
  every selected question belonging to the document;
- keeps questions stateless (no answer or conversation leaks into the next row);
- enables the agent's evidence and visual-document tools in `tool_agent` mode;
- saves one atomic JSON checkpoint after every successful answer;
- records a failed row and continues with the rest of the benchmark by default;
- validates the model, RAG, parsing, retrieval, prompt, source PDF, and question
  fields before reusing a checkpoint;
- exports answer-only and amortized parse/index latency, token usage, tool calls,
  retrieved contexts, domains, and raw DocBench question types.

## PowerShell pilot

Run this from the repository root after configuring `.env` and, when Qwen is on
the remote server, opening the SSH tunnel in a separate PowerShell window:

```powershell
$python = ".\.venv\Scripts\python.exe"
$data = "D:\Datasets\DocBench\data"
$out = "D:\file-agent-local\docbench\qwen35-agent-pilot"

& $python .\generate_docbench.py `
  --data-dir $data `
  --output-dir $out `
  --env-file .env `
  --rag-mode tool_agent `
  --max-tool-rounds 6 `
  --top-k 8 `
  --max-tokens 2500 `
  --folder-id 0 `
  --limit 3
```

Use a new output directory for a different configuration. To continue an
interrupted run, repeat the exact command and add `--resume`:

```powershell
& $python .\generate_docbench.py `
  --data-dir $data `
  --output-dir $out `
  --env-file .env `
  --rag-mode tool_agent `
  --max-tool-rounds 6 `
  --top-k 8 `
  --max-tokens 2500 `
  --folder-id 0 `
  --limit 3 `
  --resume
```

Changing a generation parameter while keeping the same output directory is
rejected deliberately. This prevents a single result set from silently mixing
different models or RAG settings.

## Full run

After inspecting the pilot, remove the selection options and use a fresh output
directory:

```powershell
$out = "D:\file-agent-local\docbench\qwen35-agent-full"

& $python .\generate_docbench.py `
  --data-dir $data `
  --output-dir $out `
  --env-file .env `
  --rag-mode tool_agent `
  --max-tool-rounds 6 `
  --top-k 8 `
  --max-tokens 2500
```

Useful subset options are `--folder-start`, `--folder-end`, repeatable
`--folder-id`, repeatable `--record-id`, repeatable `--domain`, repeatable
`--question-type`, and `--limit`. Run `python generate_docbench.py --help` for
the complete interface.

## Outputs

A complete run publishes:

```text
answers.parquet                 all answers and diagnostics
hf_dataset/                     the same rows in datasets.load_from_disk format
predictions.jsonl               readable lossless generation records
docbench_eval_input.jsonl       question/sys_ans/answer/evidence/type for judging
summary.json                    aggregate latency and usage statistics
run_manifest.json               exact run selection and generation parameters
run_config.json                 immutable resume contract for this output directory
generation.log                  persistent UTF-8 logs
checkpoints/*.json              resumable answer checkpoints
documents/*.json                per-document parsing/indexing diagnostics
failures/*.json                 detailed errors from failed attempts
```

If one or more rows fail, the same exports are written with a `partial_` prefix,
and `incomplete_rows.json` lists what must be retried. Final unprefixed outputs
are published only when all selected rows have valid checkpoints. A successful
resume removes the stale partial exports but retains the checkpoints and failure
history directory for auditability.

`docbench_eval_input.jsonl` already uses the fields expected by DocBench's judge:
`question`, `sys_ans`, `answer`, `evidence`, and `type`. It also includes stable
`id`, `file`, `question_index`, and `domain` fields, avoiding the original demo
runner's fragile numbered-text parsing.
