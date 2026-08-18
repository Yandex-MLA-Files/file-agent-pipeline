import hashlib
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.agent.loop import run_react_agent
from file_agent.agent.observability import finish_trace, pipeline_trace
from file_agent.agent.tools import build_default_tools
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_generation_llm_client
from file_agent.rag import index_documents, load_documents
from file_agent.telemetry import configure_telemetry, tracer

configure_telemetry()

SUPPORTED_TYPES = ["md", "txt", "pdf", "docx", "html", "htm", "xlsx", "pptx", "jpg", "jpeg", "png"]
CHAT_TITLE_LIMIT = 40
DEFAULT_CHAT_TOP_K = 5
# User+assistant turns kept in the history sent to the agent (not the same as
# how many are shown on screen - the full history stays for display). Every
# past turn's own tool-call scratch-work already did its job producing that
# turn's answer, so only the plain Q&A pair is worth repeating on later
# turns; even so, the local vLLM context window is tight enough with
# retrieved passages that unlimited history would eventually crowd it out.
MAX_HISTORY_TURNS = 6


def _uploaded_files_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(Path(uploaded_file.name).name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def _new_chat() -> dict:
    return {
        "title": None,
        "history": [],
        "summary": None,
        "summarized_up_to": 0,
        # Each chat owns its own documents/retriever - independent in-memory
        # LanceDB instances (the default uri="memory://"), so attaching
        # different files to different chats never collides. Created eagerly
        # (empty), not lazily on first upload, so a chat with no documents
        # yet can still use search_documents (LanceDBRetriever.search()
        # returns [] before anything is indexed - no crash, just no hits).
        "files_fingerprint": None,
        "documents": [],
        "retriever": LanceDBRetriever(),
        "document_paths": {},
        "upload_dir": None,
    }


def _start_new_chat() -> str:
    chat_id = uuid.uuid4().hex
    st.session_state.setdefault("chats", {})[chat_id] = _new_chat()
    st.session_state["active_chat_id"] = chat_id
    return chat_id


def _active_chat_id() -> str:
    chats = st.session_state.setdefault("chats", {})
    active_id = st.session_state.get("active_chat_id")
    if active_id not in chats:
        active_id = _start_new_chat()
    return active_id


def _delete_chat(chat_id: str) -> None:
    chats = st.session_state.get("chats", {})
    chat = chats.pop(chat_id, None)
    if chat is not None:
        _clear_chat_documents(chat)  # releases its retriever/upload_dir too
    if st.session_state.get("active_chat_id") == chat_id:
        # Fall back to the most recently created remaining chat; if none is
        # left, clearing active_chat_id makes _active_chat_id() start a fresh
        # one on the next run, same as the very first visit.
        remaining = list(chats.keys())
        st.session_state["active_chat_id"] = remaining[-1] if remaining else None


def _clear_chat_documents(chat: dict) -> None:
    retriever = chat.get("retriever")
    if retriever is not None:
        retriever.clear()
    upload_dir = chat.get("upload_dir")
    if upload_dir is not None:
        shutil.rmtree(upload_dir, ignore_errors=True)
    chat["files_fingerprint"] = None
    chat["documents"] = []
    # A fresh instance, not None: the chat area no longer requires documents
    # to be usable, so it must always have a valid (possibly empty) retriever.
    chat["retriever"] = LanceDBRetriever()
    chat["document_paths"] = {}
    chat["upload_dir"] = None


def _ingest_chat_documents(chat: dict, uploaded_files) -> None:
    """(Re)parse and index this chat's attached files if they changed. Stops
    the whole script (st.stop) on failure, same as a fatal startup error."""
    fingerprint = _uploaded_files_fingerprint(uploaded_files)
    if chat["files_fingerprint"] == fingerprint:
        return

    upload_dir = Path(tempfile.mkdtemp(prefix="file-agent-app-"))
    file_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        file_path = upload_dir / Path(uploaded_file.name).name
        file_path.write_bytes(uploaded_file.getbuffer())
        file_paths.append(file_path)

    try:
        with (
            st.sidebar.status("Parsing and indexing...", expanded=False),
            tracer.start_as_current_span("file_agent.ingest_files"),
        ):
            documents = load_documents(file_paths)
            retriever = LanceDBRetriever()
            index_documents(documents, retriever)
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        st.error(f"Could not parse or index uploaded files: {exc}")
        st.stop()

    _clear_chat_documents(chat)  # releases this chat's previous retriever/upload_dir, if any
    chat["files_fingerprint"] = fingerprint
    chat["documents"] = documents
    chat["retriever"] = retriever
    chat["document_paths"] = {path.name: path for path in file_paths}
    chat["upload_dir"] = upload_dir
    # New documents invalidate this chat's own conversation about the old ones.
    chat["history"] = []
    chat["summary"] = None
    chat["summarized_up_to"] = 0


def _trimmed_history(history: list[dict]) -> list[dict[str, str]]:
    """Reduce stored chat turns to the plain {role, content} shape
    run_react_agent expects, capping to the most recent MAX_HISTORY_TURNS
    turns."""
    recent = history[-MAX_HISTORY_TURNS * 2 :]
    return [{"role": message["role"], "content": message["content"]} for message in recent]


def _summarize_turns(llm_client, turns: list[dict]) -> str:

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    prompt = (
        "Summarize the key facts, decisions, and open questions from this "
        "conversation excerpt in 2-3 sentences, in the same language it's "
        "written in. Be concise - this is a running memory aid, not a report.\n\n"
        f"{transcript}"
    )
    return llm_client.generate(prompt).strip()


def _update_history_summary(llm_client, chat: dict) -> str | None:

    history = chat["history"][:-1]  # exclude the question just appended for this turn
    boundary = max(0, len(history) - MAX_HISTORY_TURNS * 2)
    new_turns = history[chat["summarized_up_to"] : boundary]
    if new_turns:
        piece = _summarize_turns(llm_client, new_turns)
        chat["summary"] = f"{chat['summary']} {piece}".strip() if chat["summary"] else piece
        chat["summarized_up_to"] = boundary
    return chat["summary"]


def _generate_chat_title(llm_client, question: str, answer: str) -> str:

    fallback = question if len(question) <= CHAT_TITLE_LIMIT else question[:CHAT_TITLE_LIMIT] + "…"
    prompt = (
        "Write a short chat title (3-6 words, no quotes, no trailing "
        "punctuation) summarizing what this conversation is about, in the "
        "same language as the text below. Reply with only the title.\n\n"
        f"Question: {question}\nAnswer: {answer[:500]}"
    )
    try:
        title = llm_client.generate(prompt).strip().strip('"').strip("'")
    except Exception:
        return fallback
    if not title:
        return fallback
    return title if len(title) <= CHAT_TITLE_LIMIT else title[:CHAT_TITLE_LIMIT] + "…"


st.set_page_config(page_title="File Agent Pipeline", layout="wide")
st.title("💬 File Agent Pipeline")

active_id = _active_chat_id()
chats = st.session_state["chats"]
chat = chats[active_id]
history = chat["history"]

# Computed up front, before anything that could touch chat["history"]:
# a question is "pending" exactly when it's the last message and nobody has
# answered it yet. State lives in history itself, not a separate
# session_state flag - no new field to forget to migrate for chats already
# sitting in an open browser tab from before this existed.
has_pending_question = bool(history) and history[-1]["role"] == "user"

with st.sidebar:
    st.header("Chats")
    if st.button("➕ New chat", use_container_width=True):
        _start_new_chat()
        st.rerun()
    for chat_id, other_chat in reversed(list(chats.items())):
        switch_col, delete_col = st.columns([5, 1])
        with switch_col:
            if st.button(
                other_chat["title"] or "New chat",
                key=f"switch-{chat_id}",
                use_container_width=True,
                type="primary" if chat_id == active_id else "secondary",
            ):
                st.session_state["active_chat_id"] = chat_id
                st.rerun()
        with delete_col:
            if st.button("🗑", key=f"delete-{chat_id}", help="Delete this chat"):
                _delete_chat(chat_id)
                st.rerun()

    st.divider()
    st.header("Documents for this chat")
    if has_pending_question:
        st.caption("⏳ Answering - document upload is paused until it's done.")
    else:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key=f"uploader-{active_id}",
        )
        if uploaded_files:
            _ingest_chat_documents(chat, uploaded_files)

        if chat["documents"]:
            st.caption(
                f"📄 {len(chat['documents'])} document(s): "
                + ", ".join(document.file_name for document in chat["documents"])
            )
            if st.button("🗑 Clear documents for this chat"):
                _clear_chat_documents(chat)
                st.rerun()
        else:
            st.caption("💬 No documents attached - chat away, or upload files to ask about them.")

documents = chat["documents"]

retriever = chat.get("retriever")
if retriever is None:
    retriever = LanceDBRetriever()
    chat["retriever"] = retriever

for message in history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input(
    "Ask a question", disabled=has_pending_question, key=f"chat-input-{active_id}"
)

if question and not has_pending_question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    st.rerun()

if has_pending_question:
    pending_question = history[-1]["content"]
    needs_title = chat["title"] is None
    with st.chat_message("assistant"):
        answer_text = None
        try:
            with st.spinner("Thinking..."), pipeline_trace(pending_question):
                with tracer.start_as_current_span("file_agent.ask_question") as question_span:
                    question_span.set_attribute("file_agent.question", pending_question)
                    question_span.set_attribute("file_agent.top_k", DEFAULT_CHAT_TOP_K)
                    llm_client = create_generation_llm_client()
                    history_summary = _update_history_summary(llm_client, chat)
                    tools = build_default_tools(
                        retriever,
                        document_paths=chat["document_paths"],
                        documents=documents,
                        default_top_k=DEFAULT_CHAT_TOP_K,
                    )
                    response = run_react_agent(
                        question=pending_question,
                        llm_client=llm_client,
                        tools=tools,
                        conversation_history=_trimmed_history(history[:-1]),
                        history_summary=history_summary,
                        verify_answer=False,
                    )
                finish_trace(output=response.answer)
        except Exception as exc:
            st.error(f"Could not generate answer: {exc}")
            answer_text = f"⚠️ Could not generate answer: {exc}"
        else:
            answer_text = response.answer
            st.write(answer_text)
            st.caption(f"{response.iterations} agent iteration(s)")
            if needs_title:
                chat["title"] = _generate_chat_title(llm_client, pending_question, answer_text)

        history.append({"role": "assistant", "content": answer_text})

    st.rerun()
