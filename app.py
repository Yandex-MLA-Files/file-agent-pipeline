import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from file_agent.pipeline import parse_file


SUPPORTED_TYPES = ["md", "pdf", "html", "htm"]
TEXT_PREVIEW_LIMIT = 3000


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
        except Exception as exc:
            st.error(f"Не удалось разобрать файл: {exc}")
        else:
            extracted_text = "\n\n".join(block.text for block in document.blocks)

            st.write(f"**Имя файла:** {document.file_name}")
            st.write(f"**Тип файла:** {document.file_type}")
            st.write(f"**Количество блоков:** {len(document.blocks)}")

            st.text_area(
                "Извлечённый текст",
                value=extracted_text[:TEXT_PREVIEW_LIMIT],
                height=400,
            )
