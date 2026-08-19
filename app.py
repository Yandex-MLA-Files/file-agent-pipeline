"""Streamlit UI: a chat over the uploaded documents.

The page is chat-first: upload files in the sidebar, ask questions in the chat
box, keep asking follow-ups. Every answer stays in the conversation (a rerun
no longer wipes it), carries its sources, and — in agent mode — a full account
of the steps the agent took. Document structure and parsing details live in
the collapsible library above the chat.
"""

import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.agent import AgentSession, answer_with_agent
from file_agent.agent.passages import describe_location
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_llm_client
from file_agent.rag import answer_with_results, index_documents, load_documents
from file_agent.telemetry import configure_telemetry, resume_span, span_identity, tracer

configure_telemetry()

SUPPORTED_TYPES = ["md", "txt", "pdf", "docx", "html", "htm", "xlsx", "pptx"]
TEXT_PREVIEW_LIMIT = 4000
OBSERVATION_PREVIEW_LIMIT = 2500
SOURCE_PREVIEW_LIMIT = 2500
MODE_AGENT = "Agent (multi-step)"
MODE_RAG = "Single-pass RAG"
RETRIEVAL_STATE_KEYS = (
    "indexed_files_fingerprint",
    "indexed_documents",
    "indexed_chunks",
    "lancedb_retriever",
    "ingest_span",
    "ingest_stats",
    "agent_session",
    "conversation",
)

st.set_page_config(
    page_title="File Agent",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Keep long tool output and passages readable */
      .stCode pre, .stCode code { white-space: pre-wrap !important; word-break: break-word; }
      /* Tighter chat bubbles */
      [data-testid="stChatMessage"] { padding: 0.35rem 0; }
      /* Source cards */
      .fa-source {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.5rem;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
      }
      .fa-source-head {
        font-family: "Source Code Pro", monospace;
        font-size: 0.78rem;
        opacity: 0.75;
        margin-bottom: 0.35rem;
      }
      .fa-source-text { white-space: pre-wrap; word-break: break-word; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------


def _uploaded_files_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(Path(uploaded_file.name).name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def _clear_retrieval_state() -> None:
    retriever = st.session_state.get("lancedb_retriever")
    if retriever is not None:
        retriever.clear()
    for key in RETRIEVAL_STATE_KEYS:
        st.session_state.pop(key, None)


def _reset_dialog() -> None:
    st.session_state["conversation"] = []
    session = st.session_state.get("agent_session")
    if session is not None:
        session.clear()


def _ingest(uploaded_files, files_fingerprint: str) -> None:
    """Parse and index the upload, with per-file progress in the UI."""
    started = time.perf_counter()
    with (
        st.status(f"Indexing {len(uploaded_files)} file(s)...", expanded=True) as status,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        file_paths: list[Path] = []
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / Path(uploaded_file.name).name
            file_path.write_bytes(uploaded_file.getbuffer())
            file_paths.append(file_path)

        documents = []
        try:
            with tracer.start_as_current_span("file_agent.ingest_files") as ingest_span:
                for file_path in file_paths:
                    file_started = time.perf_counter()
                    status.write(f"Parsing **{file_path.name}** ...")
                    documents.extend(load_documents([file_path]))
                    status.write(
                        f"&nbsp;&nbsp;done in {time.perf_counter() - file_started:.1f} s "
                        f"({len(documents[-1].blocks)} blocks)"
                    )
                status.write("Building the search index (BM25 + embeddings) ...")
                retriever = LanceDBRetriever()
                chunks = index_documents(documents, retriever)
                ingest_span_identity = span_identity(ingest_span)
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.error(f"Could not parse or index the uploaded files: {exc}")
            st.stop()

        elapsed = time.perf_counter() - started
        status.update(
            label=f"Indexed {len(documents)} file(s), {len(chunks)} chunks in {elapsed:.1f} s",
            state="complete",
            expanded=False,
        )

    previous_retriever = st.session_state.get("lancedb_retriever")
    if previous_retriever is not None:
        previous_retriever.clear()

    st.session_state["indexed_files_fingerprint"] = files_fingerprint
    st.session_state["indexed_documents"] = documents
    st.session_state["indexed_chunks"] = chunks
    st.session_state["lancedb_retriever"] = retriever
    st.session_state["ingest_span"] = ingest_span_identity
    st.session_state["ingest_stats"] = {"seconds": elapsed}
    st.session_state.pop("agent_session", None)
    st.session_state["conversation"] = []


# ---------------------------------------------------------------------------
# answering
# ---------------------------------------------------------------------------


def _source_view(result, citation_id: str | None = None) -> dict[str, Any]:
    """A rendering-friendly snapshot of one source passage."""
    metadata = dict(result.chunk.metadata)
    passage = metadata.pop("context", None) or result.chunk.text
    head = [f"file={metadata.get('source_file') or 'unknown'}"]
    head.extend(describe_location(metadata))
    if metadata.get("tool"):
        head.append(f"via {metadata['tool']}")
    if citation_id:
        head.insert(0, citation_id)
    elif result.score:
        head.append(f"score={result.score:g}")
    return {"head": " | ".join(head), "text": passage}


def _answer(question: str, mode: str, top_k: int) -> dict[str, Any]:
    """Run one question through the selected mode and snapshot the result."""
    documents = st.session_state["indexed_documents"]
    chunks = st.session_state["indexed_chunks"]
    retriever = st.session_state["lancedb_retriever"]
    started = time.perf_counter()

    entry: dict[str, Any] = {"role": "assistant", "mode": mode}
    try:
        with (
            resume_span(st.session_state.get("ingest_span")),
            tracer.start_as_current_span("file_agent.ask_question") as question_span,
        ):
            question_span.set_attribute("file_agent.question", question)
            question_span.set_attribute("file_agent.top_k", top_k)
            question_span.set_attribute("file_agent.answer_mode", mode)
            if mode == MODE_AGENT:
                session = st.session_state.setdefault("agent_session", AgentSession())
                response = answer_with_agent(
                    question=question,
                    llm_client=create_llm_client(),
                    retriever=retriever,
                    documents=documents,
                    session=session,
                )
                citations = list(response.citations)
                entry.update(
                    content=response.answer,
                    citations=citations,
                    fallback=response.fallback_used,
                    steps=[
                        {
                            "thought": step.thought,
                            "calls": [
                                {"tool": call.get("tool"), "arguments": call.get("arguments")}
                                for call in (
                                    step.actions
                                    or (
                                        [{"tool": step.tool, "arguments": step.arguments}]
                                        if step.tool
                                        else []
                                    )
                                )
                            ],
                            "observation": step.observation,
                            "seconds": step.elapsed_seconds,
                        }
                        for step in response.steps
                    ],
                    sources=[
                        _source_view(result, citations[i] if i < len(citations) else None)
                        for i, result in enumerate(response.sources)
                    ],
                )
            else:
                results = retriever.search(query=question, top_k=top_k)
                response = answer_with_results(
                    question=question,
                    results=results,
                    llm_client=create_llm_client(),
                    documents_count=len(documents),
                    chunks_count=len(chunks),
                )
                entry.update(
                    content=response.answer,
                    citations=[],
                    steps=[],
                    sources=[_source_view(result) for result in response.sources],
                )
    except Exception as exc:  # noqa: BLE001 - shown to the user in the chat
        entry.update(content=f"Could not generate an answer: {exc}", error=True)

    entry["seconds"] = time.perf_counter() - started
    return entry


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _render_assistant(entry: dict[str, Any], message_index: int) -> None:
    if entry.get("error"):
        st.error(entry["content"])
        return

    st.markdown(entry["content"])

    badges = [
        "🤖 agent" if entry.get("mode") == MODE_AGENT else "⚡ single-pass RAG",
        f"{entry.get('seconds', 0):.1f} s",
    ]
    if entry.get("citations"):
        badges.append("cited " + ", ".join(entry["citations"]))
    if entry.get("fallback"):
        badges.append("fell back to single-pass RAG")
    st.caption(" · ".join(badges))

    steps = entry.get("steps") or []
    if steps:
        tool_calls = sum(len(step["calls"]) for step in steps)
        step_label = f"How this was answered — {tool_calls} tool call(s), {len(steps)} step(s)"
        with st.expander(step_label):
            for step_number, step in enumerate(steps, start=1):
                seconds = f" · {step['seconds']:.1f} s" if step.get("seconds") else ""
                if step["calls"]:
                    for call in step["calls"]:
                        st.markdown(f"**Step {step_number}**{seconds} — `{call['tool']}`")
                        if call.get("arguments"):
                            st.code(_format_arguments(call["arguments"]), language="text")
                else:
                    label = "final answer" if step_number == len(steps) else "no tool call"
                    st.markdown(f"**Step {step_number}**{seconds} — {label}")
                if step.get("thought"):
                    st.caption(step["thought"])
                if step.get("observation"):
                    observation = step["observation"]
                    if len(observation) > OBSERVATION_PREVIEW_LIMIT:
                        observation = observation[:OBSERVATION_PREVIEW_LIMIT] + "\n[...]"
                    st.code(observation, language="text")

    sources = entry.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for source in sources:
                text = source["text"]
                if len(text) > SOURCE_PREVIEW_LIMIT:
                    text = text[:SOURCE_PREVIEW_LIMIT] + " [...]"
                st.markdown(
                    '<div class="fa-source">'
                    f'<div class="fa-source-head">{_escape(source["head"])}</div>'
                    f'<div class="fa-source-text">{_escape(text)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )


def _format_arguments(arguments: dict[str, Any]) -> str:
    parts = []
    for name, value in arguments.items():
        rendered = str(value)
        if len(rendered) > 300:
            rendered = rendered[:300] + "..."
        parts.append(f"{name}={rendered}")
    return "\n".join(parts) if parts else "(no arguments)"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_library(documents, chunks) -> None:
    total_pages = sum(document.metadata.get("total_pages", 0) for document in documents)
    with st.expander(
        f"📚 Library — {len(documents)} file(s), {total_pages} page(s), {len(chunks)} chunks",
        expanded=False,
    ):
        tabs = st.tabs([document.file_name for document in documents])
        for tab, document in zip(tabs, documents, strict=True):
            with tab:
                metadata = document.metadata
                analysis = metadata.get("page_analysis") or {}
                ocr_pages = analysis.get("ocr_page_numbers") or []
                facts = st.columns(4)
                facts[0].metric("Pages", metadata.get("total_pages", 0))
                facts[1].metric("Blocks", len(document.blocks))
                facts[2].metric("Headings", len(metadata.get("table_of_contents") or []))
                facts[3].metric("OCR pages", len(ocr_pages))
                details = [f"parsed with `{metadata.get('parsing_method', 'unknown')}`"]
                if metadata.get("ocr_engine"):
                    details.append(f"OCR engine `{metadata['ocr_engine']}`")
                if metadata.get("vlm_described_figures"):
                    described = metadata["vlm_described_figures"]
                    details.append(f"{described} figure(s) described by VLM")
                st.caption(" · ".join(details))

                toc = metadata.get("table_of_contents") or []
                if toc:
                    lines = []
                    for heading in toc[:80]:
                        indent = "&nbsp;" * 4 * max(int(heading.get("level") or 1) - 1, 0)
                        page = f" · p. {heading['page']}" if heading.get("page") else ""
                        lines.append(f"{indent}{_escape(heading['title'])}{page}")
                    if len(toc) > 80:
                        lines.append(f"... and {len(toc) - 80} more")
                    st.markdown(
                        "<div style='font-size:0.85rem; line-height:1.7'>"
                        + "<br>".join(lines)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                preview = "\n\n".join(block.text for block in document.blocks if block.text.strip())
                st.text_area(
                    "Text preview",
                    value=preview[:TEXT_PREVIEW_LIMIT],
                    height=220,
                    key=f"preview-{document.file_name}",
                    label_visibility="collapsed",
                )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 File Agent")
    st.caption("Ask questions about your documents; answers cite their sources.")

    uploaded_files = st.file_uploader(
        "Documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, XLSX, Markdown, HTML and plain text.",
    )

    st.subheader("Settings")
    answer_mode = st.radio(
        "Answer mode",
        options=[MODE_AGENT, MODE_RAG],
        help=(
            "The agent plans its own tool calls — search, exact text lookup, "
            "reading sections and pages, computing over tables — and cites the "
            "passages it used. Single-pass RAG answers from one retrieval."
        ),
    )
    top_k = st.slider(
        "Passages per retrieval",
        min_value=1,
        max_value=20,
        value=5,
        help="How many passages one retrieval returns (single-pass RAG reads exactly these).",
    )

    conversation = st.session_state.get("conversation") or []
    if conversation:
        st.divider()
        turns = sum(1 for message in conversation if message["role"] == "user")
        st.caption(f"Dialog: {turns} question(s). Follow-ups can refer to earlier answers.")
        if st.button("🧹 New dialog", use_container_width=True):
            _reset_dialog()
            st.rerun()

# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------

if not uploaded_files:
    _clear_retrieval_state()
    st.title("Chat with your documents")
    st.markdown(
        "Upload one or several files in the sidebar — reports, lectures, "
        "clinical guidelines, spreadsheets — and ask questions in any language.\n\n"
        "- **Agent mode** searches, reads sections and computes over tables, then "
        "answers with citations.\n"
        "- **Single-pass RAG** answers from one retrieval — faster, simpler questions.\n"
    )
    st.info("Waiting for files. Parsing runs once per upload; questions are instant after that.")
    st.stop()

files_fingerprint = _uploaded_files_fingerprint(uploaded_files)
if st.session_state.get("indexed_files_fingerprint") != files_fingerprint:
    _ingest(uploaded_files, files_fingerprint)

documents = st.session_state["indexed_documents"]
chunks = st.session_state["indexed_chunks"]

_render_library(documents, chunks)

conversation = st.session_state.setdefault("conversation", [])
for message_index, message in enumerate(conversation):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            _render_assistant(message, message_index)

prompt = st.chat_input("Ask about the documents...")
if prompt and prompt.strip():
    question = prompt.strip()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        working = (
            "Planning, searching and reading the documents..."
            if answer_mode == MODE_AGENT
            else "Retrieving passages and answering..."
        )
        with st.spinner(working):
            entry = _answer(question, answer_mode, int(top_k))
        _render_assistant(entry, len(conversation) + 1)
    conversation.append({"role": "user", "content": question})
    conversation.append(entry)
    # Re-render from state so the sidebar dialog counter and the "New dialog"
    # button pick up this turn immediately.
    st.rerun()
