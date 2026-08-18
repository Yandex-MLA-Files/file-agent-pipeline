"""One-off run: run our ReAct agent over every DocBench folder (229 documents,
~1074 questions) and write an answers.parquet that eval_pipeline/scripts/
run_eval.py can score as-is. v2: questions within a folder run concurrently
(MAX_CONCURRENT_QUESTIONS), and tools are rebuilt per question instead of once
per folder (see _answer_one_question). The original sequential v1 run's
~239+ already-answered questions are preserved untouched in
runs/docbench-pilot-001(-backup)/ - this writes to a fresh docbench-pilot-002
directory rather than reusing/clearing v1's, precisely so that comparison
stays possible and nothing already computed is put at risk.

DocBench (github.com/Anni-Zou/DocBench) is an external benchmark, borrowed
here only for its (question, PDF, gold answer) triples - scoring stays on our
own existing RagasJudge (DeepSeek-V4-Flash), not DocBench's own GPT-4 grader.
Not part of the library; data/<n>/<n>_qa.jsonl + data/<n>/*.pdf is expected to
already be unpacked locally (see data/.gitignore - not committed).

Usage (needs a configured file-agent-pipeline/.env and a reachable LLM
endpoint, e.g. the ssh tunnel to the shared cluster):
    uv run python scripts/docbench_pilot.py
Safe to Ctrl+C and rerun the same command - already-answered questions are
loaded from runs/docbench-pilot-002/checkpoints/ instead of recomputed, same
idea as hf_batch.py's per-row checkpoints (deliberately simpler: no schema
versioning/parameter-fingerprint matching, this isn't a shared production
artifact - delete the checkpoints dir if this script's own code changes).

Then score it from the eval_pipeline venv:
    cd eval_pipeline && uv run python scripts/run_eval.py \
        --run ../runs/docbench-pilot-002/answers.parquet \
        --out reports/docbench-pilot-002
"""

import concurrent.futures
import contextvars
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import json  # noqa: E402

import pandas as pd  # noqa: E402
from langfuse import propagate_attributes  # noqa: E402

from file_agent.agent.loop import run_react_agent  # noqa: E402
from file_agent.agent.observability import pipeline_trace  # noqa: E402
from file_agent.agent.tools import build_default_tools  # noqa: E402
from file_agent.lancedb_retriever import LanceDBRetriever  # noqa: E402
from file_agent.llm.factory import create_generation_llm_client  # noqa: E402
from file_agent.rag import index_documents, load_documents  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RUN_DIR = REPO_ROOT / "runs" / "docbench-pilot-002"
OUTPUT_PATH = RUN_DIR / "answers.parquet"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"

# una-web (needs live web search, no tool for that) is skipped everywhere,
# not just here.
SKIPPED_TYPES = {"una-web"}

# A single question is capped at this long - observed some questions pushing
# a huge read_page dump into context and then taking 10+ minutes for a 27B
# model to answer over a possibly-loaded shared endpoint. The underlying
# OpenAI client already retries up to ~16 minutes worst case (240s timeout x
# up to 4 attempts, see llm/factory.py); this is a separate, shorter, whole-
# question wall-clock cap so one slow question can't stall the whole run.
QUESTION_TIMEOUT_SECONDS = 4 * 60

# Raised from 3 to 8 once concurrency was confirmed actually working (see
# overlapping trace timestamps). Still not unlimited - the LLM endpoint is a
# shared tunnel other teammates also use, not a resource to monopolize.
MAX_CONCURRENT_QUESTIONS = 8


def _discover_all_folders() -> list[str]:
    return sorted(
        (path.name for path in DATA_DIR.iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    )


def _load_folder(folder_id: str) -> tuple[Path, list[dict]]:
    folder = DATA_DIR / folder_id
    pdfs = list(folder.glob("*.pdf"))
    if len(pdfs) != 1:
        raise RuntimeError(f"expected exactly one PDF in {folder}, found {len(pdfs)}")

    rows = []
    with (folder / f"{folder_id}_qa.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pdfs[0], rows


def _checkpoint_path(question_id: str) -> Path:
    return CHECKPOINTS_DIR / f"{question_id}.json"


def _load_checkpoint(question_id: str) -> dict | None:
    path = _checkpoint_path(question_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(row: dict) -> None:
    # Write-to-temp-then-replace so a crash mid-write never leaves a half
    # -written checkpoint that would poison the next resume. Each question
    # writes only its own uniquely-named file, so this is safe to call from
    # several worker threads at once - no shared file, no shared lock needed.
    path = _checkpoint_path(row["id"])
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _backfill_checkpoints_from_existing_output() -> None:
    """One-time migration: if answers.parquet already has rows from a run
    before checkpointing existed, seed checkpoints from it so a restart
    doesn't recompute work that's already sitting on disk."""
    if not OUTPUT_PATH.exists():
        return
    for row in pd.read_parquet(OUTPUT_PATH).to_dict("records"):
        if _checkpoint_path(row["id"]).exists():
            continue
        row["contexts"] = list(row["contexts"])  # parquet round-trips lists as numpy arrays
        _write_checkpoint(row)


def _answer_with_timeout(question: str, llm_client, tools):
    # run_react_agent has no timeout of its own; ThreadPoolExecutor lets us
    # give up waiting on the calling side without needing to actually kill
    # the underlying blocking network call - the orphaned thread just finishes
    # on its own later and its result is discarded (shutdown(wait=False), not
    # the default wait=True, which would block right back until it's done).
    # contextvars.copy_context() + ctx.run(...), not a plain submit(fn, ...):
    # OTel's (and therefore Langfuse's) current-span/baggage tracking lives in
    # contextvars, which a new thread does NOT inherit by default - without
    # this, run_react_agent would execute with no active span/session_id
    # regardless of the propagate_attributes()/pipeline_trace() wrapping it in
    # the calling thread (see _answer_one_question).
    ctx = contextvars.copy_context()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            ctx.run, run_react_agent, question, llm_client, tools, verify_answer=False
        )
        return future.result(timeout=QUESTION_TIMEOUT_SECONDS)
    finally:
        executor.shutdown(wait=False)


def _answer_one_question(
    folder_id: str,
    index: int,
    qa: dict,
    llm_client,
    retriever,
    document_paths: dict,
    documents,
) -> dict:
    # Built fresh per question, not once per folder and reused - a shared
    # tools object's accumulated_context (what run_python sees via
    # /data/_context.json) would otherwise carry search_documents/read_page
    # results from earlier questions in the same folder into a later,
    # unrelated question. app.py already rebuilds tools every turn; this
    # script didn't until now, and it would have been an outright race once
    # questions started running concurrently below.
    tools = build_default_tools(retriever, document_paths=document_paths, documents=documents)

    question_started_at = time.monotonic()

    # A genuinely flaky or slow call must not take the whole run down (or
    # stall it) with it - record it as a failed answer and keep going, same
    # posture as app.py's own exception handling around run_react_agent.
    #
    # propagate_attributes groups every span this question creates (and
    # anything nested under it, once ctx is carried across into
    # _answer_with_timeout's inner thread) under one Langfuse session -
    # RUN_DIR.name so all of v2's traces show up together, distinct from v1's
    # or any other work. pipeline_trace opens the actual root span; without
    # it, run_react_agent's own internal finish_trace() call has nothing
    # active to update (that's the "No active span" warning seen in every
    # run so far - this fixes it as a side effect, not just adds session_id).
    try:
        with propagate_attributes(session_id=RUN_DIR.name, tags=["docbench"]):
            with pipeline_trace(qa["question"]):
                response = _answer_with_timeout(qa["question"], llm_client, tools)
        answer_model = response.answer
        contexts = [source.chunk.text for source in response.sources]
        iterations = response.iterations
    except concurrent.futures.TimeoutError:
        answer_model = f"ERROR: timed out after {QUESTION_TIMEOUT_SECONDS}s"
        contexts = []
        iterations = "timeout"
    except Exception as exc:
        answer_model = f"ERROR: {exc}"
        contexts = []
        iterations = "-"

    row = {
        "id": f"docbench-{folder_id}-{index}",
        "question": qa["question"],
        "answer_model": answer_model,
        "contexts": contexts,
        "answer": qa["answer"],
        # Extra columns beyond eval_pipeline's REQUIRED_COLUMNS - carried
        # through untouched by RagasJudge.evaluate(), useful for a
        # per-type breakdown afterward.
        "type": qa["type"],
        "folder": folder_id,
    }
    _write_checkpoint(row)
    question_elapsed = time.monotonic() - question_started_at
    # Printed from whichever worker thread finishes first - lines from
    # concurrently-running questions interleave in completion order, not
    # submission order. Expected, not a bug. The time here is this one
    # question's own wall-clock, not affected by how many others were running
    # alongside it - overlapping times across lines is what confirms
    # concurrency, same as the Langfuse trace check.
    print(f"  [{qa['type']:12s}] {iterations} it. {question_elapsed:5.1f}s | {qa['question'][:70]}")
    return row


def main() -> None:
    # monotonic, not time.time(): wall-clock-only, immune to any system clock
    # adjustment during a run that can take a long time (many folders x LLM
    # calls with retries).
    started_at = time.monotonic()
    llm_client = create_generation_llm_client(env_file=REPO_ROOT / ".env")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _backfill_checkpoints_from_existing_output()

    folders = _discover_all_folders()
    print(f"Discovered {len(folders)} folders under {DATA_DIR}")

    result_rows = []
    for folder_id in folders:
        pdf_path, qa_rows = _load_folder(folder_id)
        answerable = [(i, qa) for i, qa in enumerate(qa_rows) if qa["type"] not in SKIPPED_TYPES]
        cached = {i: _load_checkpoint(f"docbench-{folder_id}-{i}") for i, _ in answerable}
        cached = {i: row for i, row in cached.items() if row is not None}

        if len(cached) == len(answerable):
            print(f"=== folder {folder_id}: {pdf_path.name} - already checkpointed, skipping ===")
            result_rows.extend(cached[i] for i, _ in answerable)
            continue

        print(f"=== folder {folder_id}: {pdf_path.name} ({len(qa_rows)} questions) ===")
        documents = load_documents([pdf_path])
        retriever = LanceDBRetriever()
        index_documents(documents, retriever)
        document_paths = {pdf_path.name: pdf_path}

        to_run = []
        for i, qa in answerable:
            if i in cached:
                result_rows.append(cached[i])
                print(f"  [cached] {qa['question'][:70]}")
            else:
                to_run.append((i, qa))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUESTIONS)
        with executor:
            futures = [
                executor.submit(
                    _answer_one_question,
                    folder_id,
                    i,
                    qa,
                    llm_client,
                    retriever,
                    document_paths,
                    documents,
                )
                for i, qa in to_run
            ]
            for future in concurrent.futures.as_completed(futures):
                result_rows.append(future.result())

        # Written after every folder, not only at the end - a crash partway
        # through (flaky LLM, dropped tunnel) still leaves every folder
        # completed so far on disk instead of losing the whole run.
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows).to_parquet(OUTPUT_PATH, index=False)

    elapsed = timedelta(seconds=round(time.monotonic() - started_at))
    errors = sum(1 for row in result_rows if str(row["answer_model"]).startswith("ERROR"))
    print(f"\nWrote {len(result_rows)} rows to {OUTPUT_PATH} ({errors} error(s))")
    print(f"Took {elapsed}")


if __name__ == "__main__":
    main()
