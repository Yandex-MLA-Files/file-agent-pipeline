import fitz
from PIL import Image

from file_agent.document import Block, BlockType, Document
from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.vlm.base import VLMClient


class StubVLMClient(VLMClient):
    def __init__(self, description="STUB: a diagram"):
        self.description = description
        self.calls = 0

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        return self.description


class ExplodingVLMClient(VLMClient):
    def describe_image(self, image: Image.Image, prompt: str) -> str:  # pragma: no cover
        raise AssertionError("VLM must not be called")


def _make_pdf_with_shape(path):
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 1), fill=(0, 0, 1))
    doc.save(path)
    doc.close()


def test_enhancer_describes_figure_block(tmp_path):
    pdf_path = tmp_path / "diagram.pdf"
    _make_pdf_with_shape(pdf_path)

    figure = Block(
        id="f1",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        page_number=1,
        bbox=(50.0, 50.0, 200.0, 200.0),
    )
    document = Document(file_name="diagram.pdf", file_type="pdf", blocks=[figure])

    stub = StubVLMClient()
    DocumentEnhancer(vlm_client=stub).enhance(document, pdf_path)

    assert stub.calls == 1
    assert figure.vlm_description == "STUB: a diagram"
    assert "[Image description]: STUB: a diagram" in figure.text


def test_enhancer_skips_blocks_without_bbox(tmp_path):
    pdf_path = tmp_path / "diagram.pdf"
    _make_pdf_with_shape(pdf_path)

    text_block = Block(id="t1", text="plain", type="text", block_type=BlockType.TEXT)
    figure_no_bbox = Block(
        id="f1", text="", type="figure", block_type=BlockType.FIGURE, page_number=1
    )
    document = Document(file_name="d.pdf", file_type="pdf", blocks=[text_block, figure_no_bbox])

    # ExplodingVLMClient asserts it is never invoked.
    DocumentEnhancer(vlm_client=ExplodingVLMClient()).enhance(document, pdf_path)

    assert text_block.vlm_description is None
    assert figure_no_bbox.vlm_description is None


def test_enhancer_ignores_non_pdf(tmp_path):
    docx_path = tmp_path / "doc.docx"
    docx_path.write_bytes(b"not really a docx")

    figure = Block(
        id="f1",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        page_number=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
    )
    document = Document(file_name="doc.docx", file_type="docx", blocks=[figure])

    DocumentEnhancer(vlm_client=ExplodingVLMClient()).enhance(document, docx_path)

    assert figure.vlm_description is None
