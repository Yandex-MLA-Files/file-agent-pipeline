import hashlib
import logging
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
load_dotenv(PROJECT_ROOT / ".env")

from file_agent.chat_ui import (
    DEFAULT_CHAT_TITLE,
    ChatMessage,
    ChatSession,
    CompactSource,
    chat_title_from_question,
    compact_sources,
    create_chat_session,
)
from file_agent.document_assets import InMemoryDocumentAssetStore
from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_llm_client
from file_agent.rag import (
    answer_indexed_documents,
    answer_with_results,
    ingest_files,
    resolve_rag_mode,
)
from file_agent.telemetry import configure_telemetry, resume_span, span_identity, tracer
from file_agent.vlm.factory import create_vlm_client

configure_telemetry()
logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ["md", "txt", "pdf", "docx", "html", "htm", "xlsx", "pptx"]
DEFAULT_TOP_K = 5
RETRIEVAL_STATE_KEYS = (
    "indexed_files_fingerprint",
    "indexed_documents",
    "indexed_chunks",
    "lancedb_retriever",
    "document_asset_store",
    "vlm_client",
    "ingest_span",
)
CHAT_SESSIONS_KEY = "chat_sessions"
ACTIVE_CHAT_KEY = "active_chat_id"
LEGACY_CHAT_THREAD_KEY = "chat_thread_id"
LEGACY_CHAT_MESSAGES_KEY = "chat_messages"


def _uploaded_files_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    file_entries = sorted(
        (Path(uploaded_file.name).name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    )
    for file_name, contents in file_entries:
        digest.update(file_name.encode("utf-8"))
        digest.update(contents)
    return digest.hexdigest()


def _create_new_chat() -> str:
    thread_id = uuid.uuid4().hex
    sessions = st.session_state.setdefault(CHAT_SESSIONS_KEY, {})
    sessions[thread_id] = create_chat_session(thread_id)
    st.session_state[ACTIVE_CHAT_KEY] = thread_id
    return thread_id


def _reset_chat_state() -> None:
    st.session_state[CHAT_SESSIONS_KEY] = {}
    st.session_state.pop(ACTIVE_CHAT_KEY, None)
    st.session_state.pop(LEGACY_CHAT_THREAD_KEY, None)
    st.session_state.pop(LEGACY_CHAT_MESSAGES_KEY, None)
    _create_new_chat()


def _ensure_chat_state() -> None:
    sessions = st.session_state.get(CHAT_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        legacy_thread_id = st.session_state.pop(LEGACY_CHAT_THREAD_KEY, None)
        legacy_messages = st.session_state.pop(LEGACY_CHAT_MESSAGES_KEY, [])
        thread_id = legacy_thread_id or uuid.uuid4().hex
        session = create_chat_session(thread_id)
        if isinstance(legacy_messages, list):
            session["messages"] = legacy_messages
        sessions = {thread_id: session}
        st.session_state[CHAT_SESSIONS_KEY] = sessions
        st.session_state[ACTIVE_CHAT_KEY] = thread_id

    active_chat_id = st.session_state.get(ACTIVE_CHAT_KEY)
    if active_chat_id not in sessions:
        _create_new_chat()


def _active_chat() -> ChatSession:
    _ensure_chat_state()
    return st.session_state[CHAT_SESSIONS_KEY][st.session_state[ACTIVE_CHAT_KEY]]


def _clear_retrieval_state() -> bool:
    had_indexed_documents = "indexed_files_fingerprint" in st.session_state
    retriever = st.session_state.get("lancedb_retriever")
    if retriever is not None:
        retriever.clear()
    asset_store = st.session_state.get("document_asset_store")
    if asset_store is not None:
        asset_store.clear()

    for key in RETRIEVAL_STATE_KEYS:
        st.session_state.pop(key, None)
    return had_indexed_documents


def _render_sources(sources: list[CompactSource]) -> None:
    if not sources:
        return

    with st.expander(f"Источники · {len(sources)}"):
        for index, source in enumerate(sources, start=1):
            location = f" · {source['location']}" if source.get("location") else ""
            st.markdown(f"**{index}. {source['file_name']}**{location}")
            if source.get("excerpt"):
                st.caption(source["excerpt"])


def _render_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))


def _render_chat_navigation(documents) -> None:
    sessions: dict[str, ChatSession] = st.session_state[CHAT_SESSIONS_KEY]
    active_chat_id = st.session_state[ACTIVE_CHAT_KEY]

    with st.sidebar:
        st.divider()
        if st.button("＋ Новый чат", type="primary", use_container_width=True):
            _create_new_chat()
            st.rerun()

        st.caption("ЧАТЫ")
        for thread_id, session in reversed(list(sessions.items())):
            if st.button(
                session["title"],
                key=f"open_chat_{thread_id}",
                type="primary" if thread_id == active_chat_id else "secondary",
                use_container_width=True,
            ):
                st.session_state[ACTIVE_CHAT_KEY] = thread_id
                st.rerun()

        st.divider()
        st.caption(f"ДОКУМЕНТЫ · {len(documents)}")
        for document in documents:
            st.markdown(f"📄 `{document.file_name}`")


st.set_page_config(
    page_title="File Agent",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)
_ensure_chat_state()

with st.sidebar:
    st.title("File Agent")
    st.caption("Ответы по вашим документам")
    uploaded_files = st.file_uploader(
        "Документы",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, XLSX, Markdown, TXT или HTML",
    )

if not uploaded_files:
    if _clear_retrieval_state():
        _reset_chat_state()

    st.title("Диалог с документами")
    st.write(
        "Загрузите один или несколько файлов — агент найдёт нужные фрагменты, "
        "прочитает таблицы и поможет разобраться в содержимом."
    )
    with st.container(border=True):
        st.markdown("**Как начать**")
        st.markdown("1. Загрузите документы в боковой панели.\n2. Задайте вопрос в чате.")
    st.stop()

files_fingerprint = _uploaded_files_fingerprint(uploaded_files)

if (
    st.session_state.get("indexed_files_fingerprint") != files_fingerprint
    or "document_asset_store" not in st.session_state
    or "vlm_client" not in st.session_state
):
    try:
        with st.spinner("Подготавливаем документы…"):
            asset_store = InMemoryDocumentAssetStore()
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths: list[Path] = []
                for uploaded_file in uploaded_files:
                    file_name = Path(uploaded_file.name).name
                    contents = uploaded_file.getvalue()
                    asset_store.put(file_name, contents)
                    file_path = Path(temp_dir) / file_name
                    file_path.write_bytes(contents)
                    file_paths.append(file_path)

                with tracer.start_as_current_span("file_agent.ingest_files") as ingest_span:
                    retriever = LanceDBRetriever()
                    documents, chunks = ingest_files(file_paths, retriever)
                    ingest_span_identity = span_identity(ingest_span)
            rag_mode = resolve_rag_mode()
            vlm_client = create_vlm_client() if rag_mode == "tool_agent" else None
    except Exception:
        logger.exception("Could not parse or index uploaded files")
        st.error("Не удалось обработать документы. Проверьте формат файлов и попробуйте ещё раз.")
        st.stop()

    previous_retriever = st.session_state.get("lancedb_retriever")
    if previous_retriever is not None:
        previous_retriever.clear()
    previous_asset_store = st.session_state.get("document_asset_store")
    if previous_asset_store is not None:
        previous_asset_store.clear()

    st.session_state["indexed_files_fingerprint"] = files_fingerprint
    st.session_state["indexed_documents"] = documents
    st.session_state["indexed_chunks"] = chunks
    st.session_state["lancedb_retriever"] = retriever
    st.session_state["document_asset_store"] = asset_store
    st.session_state["vlm_client"] = vlm_client
    st.session_state["ingest_span"] = ingest_span_identity
    _reset_chat_state()

documents = st.session_state["indexed_documents"]
chunks = st.session_state["indexed_chunks"]
retriever = st.session_state["lancedb_retriever"]
asset_store = st.session_state["document_asset_store"]
vlm_client = st.session_state["vlm_client"]
rag_mode = resolve_rag_mode()

_render_chat_navigation(documents)
active_chat = _active_chat()

st.title(active_chat["title"])
file_names = ", ".join(document.file_name for document in documents)
st.caption(f"Документов: {len(documents)} · {file_names}")

if not active_chat["messages"]:
    with st.container(border=True):
        st.markdown("**Документы готовы**")
        st.write("Задайте вопрос, попросите сравнить данные или объяснить таблицу или график.")

for message in active_chat["messages"]:
    _render_message(message)

query = st.chat_input("Задайте вопрос по документам")
if query:
    normalized_query = query.strip()
    if not normalized_query:
        st.stop()

    if active_chat["title"] == DEFAULT_CHAT_TITLE:
        active_chat["title"] = chat_title_from_question(normalized_query)

    user_message: ChatMessage = {"role": "user", "content": normalized_query}
    active_chat["messages"].append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with (
                st.spinner("Ищу ответ в документах…"),
                resume_span(st.session_state.get("ingest_span")),
                tracer.start_as_current_span("file_agent.ask_question") as question_span,
            ):
                question_span.set_attribute("file_agent.question", normalized_query)
                question_span.set_attribute("file_agent.top_k", DEFAULT_TOP_K)
                question_span.set_attribute("file_agent.thread_id", active_chat["thread_id"])
                llm_client = create_llm_client(load_env=False)
                if rag_mode == "tool_agent":
                    response = answer_indexed_documents(
                        question=normalized_query,
                        llm_client=llm_client,
                        retriever=retriever,
                        documents_count=len(documents),
                        chunks_count=len(chunks),
                        top_k=DEFAULT_TOP_K,
                        documents=documents,
                        mode=rag_mode,
                        thread_id=active_chat["thread_id"],
                        vlm_client=vlm_client,
                        asset_store=asset_store,
                    )
                else:
                    results = retriever.search(query=normalized_query, top_k=DEFAULT_TOP_K)
                    response = answer_with_results(
                        question=normalized_query,
                        results=results,
                        llm_client=llm_client,
                        documents_count=len(documents),
                        chunks_count=len(chunks),
                        mode=rag_mode,
                    )
        except Exception:
            logger.exception("Could not generate an answer")
            st.error(
                "Не удалось получить ответ. Проверьте подключение к модели и повторите запрос."
            )
        else:
            sources = compact_sources(response.sources)
            st.markdown(response.answer)
            _render_sources(sources)
            assistant_message: ChatMessage = {
                "role": "assistant",
                "content": response.answer,
                "sources": sources,
            }
            active_chat["messages"].append(assistant_message)
            st.rerun()
