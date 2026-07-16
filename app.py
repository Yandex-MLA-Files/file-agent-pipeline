import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.lancedb_retriever import LanceDBRetriever
from file_agent.llm.factory import create_llm_client
from file_agent.rag import (
    answer_with_results,
    index_documents,
    load_documents,
)
from file_agent.telemetry import configure_telemetry

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
    "search_cache_key",
    "search_results",
)


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


st.set_page_config(page_title="File Agent Pipeline")
st.title("File Agent Pipeline")

uploaded_files = st.file_uploader(
    "Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if not uploaded_files:
    _clear_retrieval_state()
else:
    files_fingerprint = _uploaded_files_fingerprint(uploaded_files)

    if st.session_state.get("indexed_files_fingerprint") != files_fingerprint:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths: list[Path] = []
            for uploaded_file in uploaded_files:
                file_path = Path(temp_dir) / Path(uploaded_file.name).name
                file_path.write_bytes(uploaded_file.getbuffer())
                file_paths.append(file_path)

            try:
                documents = load_documents(file_paths)
                retriever = LanceDBRetriever()
                chunks = index_documents(documents, retriever)
            except Exception as exc:
                st.error(f"Could not parse or index uploaded files: {exc}")
                st.stop()

        previous_retriever = st.session_state.get("lancedb_retriever")
        if previous_retriever is not None:
            previous_retriever.clear()

        st.session_state["indexed_files_fingerprint"] = files_fingerprint
        st.session_state["indexed_documents"] = documents
        st.session_state["indexed_chunks"] = chunks
        st.session_state["indexed_text"] = "\n\n".join(
            block.text for document in documents for block in document.blocks
        )
        st.session_state["lancedb_retriever"] = retriever

    documents = st.session_state["indexed_documents"]
    chunks = st.session_state["indexed_chunks"]
    extracted_text = st.session_state["indexed_text"]
    retriever = st.session_state["lancedb_retriever"]

    st.write(f"**Documents:** {len(documents)}")
    st.write("**Files:** " + ", ".join(document.file_name for document in documents))
    st.write("**Blocks:** " + str(sum(len(document.blocks) for document in documents)))
    st.write(f"**Chunks:** {len(chunks)}")

    with st.expander("Parsing details"):
        for document in documents:
            metadata = document.metadata
            analysis = metadata.get("page_analysis") or {}
            ocr_pages = analysis.get("ocr_page_numbers") or []
            st.write(f"**{document.file_name}**")
            st.write(
                f"- method: `{metadata.get('parsing_method', 'unknown')}`"
                + (
                    f" (OCR engine: `{metadata['ocr_engine']}`)"
                    if metadata.get("ocr_engine")
                    else ""
                )
            )
            st.write(f"- pages: {metadata.get('total_pages', 0)}, OCR'd pages: {len(ocr_pages)}")
            if ocr_pages:
                st.write(f"- OCR page numbers: {ocr_pages}")
            toc = metadata.get("table_of_contents") or []
            st.write(f"- headings detected: {len(toc)}")
            if metadata.get("vlm_described_figures"):
                st.write(f"- figures described by VLM: {metadata['vlm_described_figures']}")

    st.text_area(
        "Extracted text",
        value=extracted_text[:TEXT_PREVIEW_LIMIT],
        height=400,
    )

    query = st.text_input("Question")
    top_k = st.number_input(
        "Top K chunks",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
    generate_answer = st.button("Generate answer")
    results = []
    normalized_query = query.strip()

    if normalized_query:
        search_cache_key = (
            files_fingerprint,
            normalized_query,
            int(top_k),
        )
        if (
            st.session_state.get("search_cache_key") != search_cache_key
            or "search_results" not in st.session_state
        ):
            st.session_state["search_results"] = retriever.search(
                query=normalized_query,
                top_k=int(top_k),
            )
            st.session_state["search_cache_key"] = search_cache_key

        results = st.session_state["search_results"]

        if not results:
            st.info("No matching chunks found.")
        else:
            st.subheader("Search results")
            for index, result in enumerate(results, start=1):
                with st.expander(
                    f"Result {index} - score {result.score:g}",
                    expanded=index == 1,
                ):
                    metadata = dict(result.chunk.metadata)
                    passage = metadata.pop("context", None)
                    st.write("**Metadata:**")
                    st.json(metadata)
                    st.text_area(
                        "Matched chunk (what was embedded and searched)",
                        value=result.chunk.text[:CHUNK_PREVIEW_LIMIT],
                        height=160,
                        key=f"chunk-result-{index}",
                    )
                    if passage:
                        st.text_area(
                            "Passage sent to the LLM (parent section of this chunk)",
                            value=passage[: CHUNK_PREVIEW_LIMIT * 2],
                            height=240,
                            key=f"chunk-context-{index}",
                        )
                    else:
                        st.caption(
                            "This chunk already covers its whole section, so it is "
                            "sent to the LLM as is."
                        )

    if generate_answer:
        if not normalized_query:
            st.warning("Enter a question before generating an answer.")
        else:
            try:
                response = answer_with_results(
                    question=normalized_query,
                    results=results,
                    llm_client=create_llm_client(),
                    documents_count=len(documents),
                    chunks_count=len(chunks),
                )
            except Exception as exc:
                st.error(f"Could not generate answer: {exc}")
            else:
                st.subheader("Answer")
                st.write(response.answer)

                if response.sources:
                    st.subheader("Sources")
                    for index, result in enumerate(response.sources, start=1):
                        with st.expander(
                            f"Source {index} - score {result.score:g}",
                            expanded=index == 1,
                        ):
                            metadata = dict(result.chunk.metadata)
                            passage = metadata.pop("context", None)
                            st.write("**Metadata:**")
                            st.json(metadata)
                            st.text_area(
                                "Source text",
                                value=(passage or result.chunk.text)[: CHUNK_PREVIEW_LIMIT * 2],
                                height=240,
                                key=f"answer-source-{index}",
                            )
