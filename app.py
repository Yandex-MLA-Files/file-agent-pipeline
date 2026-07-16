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
    answer_indexed_documents,
    index_documents,
    load_documents,
)

SUPPORTED_TYPES = ["md", "pdf", "html", "htm", "xlsx", "pptx"]
TEXT_PREVIEW_LIMIT = 3000
CHUNK_PREVIEW_LIMIT = 1000
RETRIEVAL_STATE_KEYS = (
    "indexed_files_fingerprint",
    "indexed_documents",
    "indexed_chunks",
    "indexed_text",
    "lancedb_retriever",
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

    if query.strip():
        results = retriever.search(
            query=query,
            top_k=int(top_k),
        )

        if not results:
            st.info("No matching chunks found.")
        else:
            st.subheader("Search results")
            for index, result in enumerate(results, start=1):
                with st.expander(
                    f"Result {index} - score {result.score:g}",
                    expanded=index == 1,
                ):
                    st.write("**Metadata:**")
                    st.json(result.chunk.metadata)
                    st.text_area(
                        "Chunk text",
                        value=result.chunk.text[:CHUNK_PREVIEW_LIMIT],
                        height=240,
                        key=f"chunk-result-{index}",
                    )

    if generate_answer:
        if not query.strip():
            st.warning("Enter a question before generating an answer.")
        else:
            try:
                response = answer_indexed_documents(
                    question=query,
                    llm_client=create_llm_client(),
                    retriever=retriever,
                    documents_count=len(documents),
                    chunks_count=len(chunks),
                    top_k=int(top_k),
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
                            st.write("**Metadata:**")
                            st.json(result.chunk.metadata)
                            st.text_area(
                                "Source text",
                                value=result.chunk.text[:CHUNK_PREVIEW_LIMIT],
                                height=240,
                                key=f"answer-source-{index}",
                            )
