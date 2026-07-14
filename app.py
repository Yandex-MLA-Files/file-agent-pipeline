import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.chunking import chunk_document
from file_agent.llm.fake import FakeLLM
from file_agent.pipeline import parse_file
from file_agent.qa import answer_question_with_context
from file_agent.retrieval import search_chunks

SUPPORTED_TYPES = ["md", "pdf", "html", "htm", "xlsx", "pptx"]
TEXT_PREVIEW_LIMIT = 3000
CHUNK_PREVIEW_LIMIT = 1000


st.set_page_config(page_title="File Agent Pipeline")
st.title("File Agent Pipeline")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / Path(uploaded_file.name).name
        file_path.write_bytes(uploaded_file.getbuffer())

        try:
            document = parse_file(file_path)
            chunks = chunk_document(document)
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
        else:
            extracted_text = "\n\n".join(block.text for block in document.blocks)

            st.write(f"**File name:** {document.file_name}")
            st.write(f"**File type:** {document.file_type}")
            st.write(f"**Blocks:** {len(document.blocks)}")
            st.write(f"**Chunks:** {len(chunks)}")

            st.text_area(
                "Extracted text",
                value=extracted_text[:TEXT_PREVIEW_LIMIT],
                height=400,
            )

            query = st.text_input("Search in chunks")
            use_fake_llm = st.checkbox("Use Fake LLM")
            generate_answer = st.button("Generate answer")
            results = []

            if query.strip():
                results = search_chunks(query, chunks, top_k=5)

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
                elif not use_fake_llm:
                    st.warning("Enable Use Fake LLM to test QA without YandexGPT.")
                elif not results:
                    st.info("No matching chunks found, so FakeLLM was not called.")
                else:
                    answer = answer_question_with_context(
                        question=query,
                        results=results,
                        llm_client=FakeLLM(),
                    )
                    st.subheader("Answer")
                    st.write(answer)
