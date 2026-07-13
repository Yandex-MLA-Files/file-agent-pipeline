import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.llm.factory import create_llm_client
from file_agent.rag import answer_documents, chunk_documents, load_documents
from file_agent.retrieval import search_chunks


SUPPORTED_TYPES = ["md", "pdf", "html", "htm", "xlsx", "pptx"]
TEXT_PREVIEW_LIMIT = 3000
CHUNK_PREVIEW_LIMIT = 1000


st.set_page_config(page_title="File Agent Pipeline")
st.title("File Agent Pipeline")

uploaded_files = st.file_uploader(
    "Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths: list[Path] = []
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / Path(uploaded_file.name).name
            file_path.write_bytes(uploaded_file.getbuffer())
            file_paths.append(file_path)

        try:
            documents = load_documents(file_paths)
            chunks = chunk_documents(documents)
        except Exception as exc:
            st.error(f"Could not parse uploaded files: {exc}")
        else:
            extracted_text = "\n\n".join(
                block.text for document in documents for block in document.blocks
            )

            st.write(f"**Documents:** {len(documents)}")
            st.write(
                "**Files:** "
                + ", ".join(document.file_name for document in documents)
            )
            st.write(
                "**Blocks:** "
                + str(sum(len(document.blocks) for document in documents))
            )
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
            use_semantic = st.checkbox("Use semantic retrieval", value=True)
            generate_answer = st.button("Generate answer")
            results = []

            if query.strip():
                results = search_chunks(
                    query=query,
                    chunks=chunks,
                    top_k=int(top_k),
                    use_semantic=use_semantic,
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
                        response = answer_documents(
                            documents=documents,
                            question=query,
                            llm_client=create_llm_client(),
                            top_k=int(top_k),
                            use_semantic=use_semantic,
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

