import hashlib
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

SUPPORTED_TYPES = ["md", "txt", "pdf", "docx", "html", "htm", "xlsx", "pptx"]
TEXT_PREVIEW_LIMIT = 3000
CHUNK_PREVIEW_LIMIT = 1000
RETRIEVAL_STATE_KEYS = (
    "indexed_files_fingerprint",
    "indexed_documents",
    "indexed_chunks",
    "indexed_text",
    "lancedb_retriever",
    "document_asset_store",
    "vlm_client",
    "ingest_span",
)
CHAT_THREAD_KEY = "chat_thread_id"
CHAT_MESSAGES_KEY = "chat_messages"


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


def _start_new_chat() -> None:
    st.session_state[CHAT_THREAD_KEY] = uuid.uuid4().hex
    st.session_state[CHAT_MESSAGES_KEY] = []


def _ensure_chat_state() -> None:
    if CHAT_THREAD_KEY not in st.session_state:
        _start_new_chat()
    elif CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = []


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


def _show_response_details(response) -> None:
    if response.search_queries or response.tool_calls:
        with st.expander("RAG execution details"):
            st.write(f"Stop reason: `{response.stop_reason}`")
            if response.search_queries:
                st.write("Search queries:")
                for search_query in response.search_queries:
                    st.write(f"- {search_query}")
            if response.tool_calls:
                st.write("Tool calls:")
                for tool_call in response.tool_calls:
                    st.write(f"- `{tool_call['name']}`")
                    st.json(tool_call["arguments"])

    if response.sources:
        st.write("**Sources**")
        for index, result in enumerate(response.sources, start=1):
            with st.expander(
                f"Source {index} - score {result.score:g}",
                expanded=index == 1,
            ):
                metadata = dict(result.chunk.metadata)
                passage = metadata.pop("context", None)
                st.write("**Metadata:**")
                st.json(metadata)
                st.write("**Source text:**")
                st.text((passage or result.chunk.text)[: CHUNK_PREVIEW_LIMIT * 2])


st.set_page_config(page_title="File Agent Pipeline")
_ensure_chat_state()
st.title("File Agent Pipeline")
rag_mode = resolve_rag_mode()
st.caption(f"RAG mode: `{rag_mode}`")

uploaded_files = st.file_uploader(
    "Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if not uploaded_files:
    if _clear_retrieval_state():
        _start_new_chat()
    st.info("Upload one or more documents to start a chat.")
    st.stop()

files_fingerprint = _uploaded_files_fingerprint(uploaded_files)

if (
    st.session_state.get("indexed_files_fingerprint") != files_fingerprint
    or "document_asset_store" not in st.session_state
    or "vlm_client" not in st.session_state
):
    try:
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
        vlm_client = create_vlm_client() if rag_mode == "tool_agent" else None
    except Exception as exc:
        st.error(f"Could not parse or index uploaded files: {exc}")
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
    st.session_state["indexed_text"] = "\n\n".join(
        block.text for document in documents for block in document.blocks
    )
    st.session_state["lancedb_retriever"] = retriever
    st.session_state["document_asset_store"] = asset_store
    st.session_state["vlm_client"] = vlm_client
    st.session_state["ingest_span"] = ingest_span_identity
    _start_new_chat()

documents = st.session_state["indexed_documents"]
chunks = st.session_state["indexed_chunks"]
extracted_text = st.session_state["indexed_text"]
retriever = st.session_state["lancedb_retriever"]
asset_store = st.session_state["document_asset_store"]
vlm_client = st.session_state["vlm_client"]

summary_column, reset_column = st.columns([4, 1])
with summary_column:
    st.write(
        f"**Documents:** {len(documents)} · "
        f"**Blocks:** {sum(len(document.blocks) for document in documents)} · "
        f"**Chunks:** {len(chunks)}"
    )
    st.caption("Files: " + ", ".join(document.file_name for document in documents))
with reset_column:
    if st.button("New chat", use_container_width=True):
        _start_new_chat()
        st.rerun()

with st.expander("Document details"):
    for document in documents:
        metadata = document.metadata
        analysis = metadata.get("page_analysis") or {}
        ocr_pages = analysis.get("ocr_page_numbers") or []
        st.write(f"**{document.file_name}**")
        st.write(
            f"- method: `{metadata.get('parsing_method', 'unknown')}`"
            + (f" (OCR engine: `{metadata['ocr_engine']}`)" if metadata.get("ocr_engine") else "")
        )
        st.write(f"- pages: {metadata.get('total_pages', 0)}, OCR'd pages: {len(ocr_pages)}")
        if ocr_pages:
            st.write(f"- OCR page numbers: {ocr_pages}")
        toc = metadata.get("table_of_contents") or []
        st.write(f"- headings detected: {len(toc)}")
        if metadata.get("vlm_described_figures"):
            st.write(f"- figures described by VLM: {metadata['vlm_described_figures']}")

    st.text_area(
        "Extracted text preview",
        value=extracted_text[:TEXT_PREVIEW_LIMIT],
        height=300,
    )

top_k = st.number_input(
    "Top K chunks",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

for message in st.session_state[CHAT_MESSAGES_KEY]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the uploaded documents")
if query:
    normalized_query = query.strip()
    if not normalized_query:
        st.stop()

    st.session_state[CHAT_MESSAGES_KEY].append({"role": "user", "content": normalized_query})
    with st.chat_message("user"):
        st.markdown(normalized_query)

    with st.chat_message("assistant"):
        try:
            with (
                st.spinner("Searching the documents..."),
                resume_span(st.session_state.get("ingest_span")),
                tracer.start_as_current_span("file_agent.ask_question") as question_span,
            ):
                question_span.set_attribute("file_agent.question", normalized_query)
                question_span.set_attribute("file_agent.top_k", int(top_k))
                question_span.set_attribute(
                    "file_agent.thread_id", st.session_state[CHAT_THREAD_KEY]
                )
                llm_client = create_llm_client(load_env=False)
                if rag_mode == "tool_agent":
                    response = answer_indexed_documents(
                        question=normalized_query,
                        llm_client=llm_client,
                        retriever=retriever,
                        documents_count=len(documents),
                        chunks_count=len(chunks),
                        top_k=int(top_k),
                        documents=documents,
                        mode=rag_mode,
                        thread_id=st.session_state[CHAT_THREAD_KEY],
                        vlm_client=vlm_client,
                        asset_store=asset_store,
                    )
                else:
                    results = retriever.search(
                        query=normalized_query,
                        top_k=int(top_k),
                    )
                    response = answer_with_results(
                        question=normalized_query,
                        results=results,
                        llm_client=llm_client,
                        documents_count=len(documents),
                        chunks_count=len(chunks),
                        mode=rag_mode,
                    )
        except Exception as exc:
            st.error(f"Could not generate answer: {exc}")
        else:
            st.markdown(response.answer)
            _show_response_details(response)
            st.session_state[CHAT_MESSAGES_KEY].append(
                {"role": "assistant", "content": response.answer}
            )
