from file_agent.parsers.enhancer import DocumentEnhancer
from file_agent.vlm.base import VLMClient
from file_agent.vlm.factory import create_vlm_client
from file_agent.vlm.openai_compatible import OpenAICompatibleVLMClient
from file_agent.vlm.smolvlm import SmolVLMClient


def test_factory_default_llm_backend_needs_the_chat_endpoint(monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    for name in ("LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL", "VLM_BASE_URL", "VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    # Default backend is the chat model's endpoint; without one, no VLM.
    assert create_vlm_client() is None

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-27B")
    monkeypatch.delenv("VLM_ENABLE_THINKING", raising=False)

    client = create_vlm_client()

    assert isinstance(client, OpenAICompatibleVLMClient)
    assert client.model == "Qwen/Qwen3.5-27B"
    assert client.enable_thinking is False


def test_factory_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "off")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-27B")

    assert create_vlm_client() is None


def test_factory_builds_local_smolvlm(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "smolvlm")
    monkeypatch.setenv("VLM_LOCAL_MODEL", "some/checkpoint")

    client = create_vlm_client()

    # Construction must be lazy: no model download happens here.
    assert isinstance(client, SmolVLMClient)
    assert client.model_name == "some/checkpoint"
    assert client._model is None


def test_factory_builds_openai_client(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.setenv("VLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VLM_MODEL", "qwen2.5-vl:7b")

    client = create_vlm_client()

    assert isinstance(client, OpenAICompatibleVLMClient)


def test_factory_openai_requires_configuration(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)

    assert create_vlm_client() is None


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "quantum")

    assert create_vlm_client() is None


class CountingVLM(VLMClient):
    def __init__(self):
        self.calls = 0

    def describe_image(self, image, prompt):
        self.calls += 1
        return f"description {self.calls}"


def test_enhancer_caps_figure_count_and_prefers_large_figures(tmp_path):
    import fitz

    from file_agent.document import Block, BlockType, Document

    pdf_path = tmp_path / "figs.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(10, 10, 500, 400), fill=(0, 0, 1))
    doc.save(pdf_path)
    doc.close()

    blocks = [
        Block(
            id=f"f{i}",
            text="",
            type="figure",
            block_type=BlockType.FIGURE,
            page_number=1,
            # Increasing sizes: f0 is smallest, f4 largest.
            bbox=(10.0, 10.0, 10.0 + 100 * (i + 1), 10.0 + 100 * (i + 1)),
        )
        for i in range(5)
    ]
    document = Document(file_name="figs.pdf", file_type="pdf", blocks=blocks)

    vlm = CountingVLM()
    DocumentEnhancer(vlm_client=vlm, max_figures=2, min_figure_area=0).enhance(document, pdf_path)

    assert vlm.calls == 2
    described = [b.id for b in blocks if b.vlm_description]
    assert described == ["f3", "f4"] or set(described) == {"f3", "f4"}
    assert document.metadata["vlm_described_figures"] == 2


def test_enhancer_skips_tiny_decorative_images(tmp_path):
    import fitz

    from file_agent.document import Block, BlockType, Document

    pdf_path = tmp_path / "icons.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    icon = Block(
        id="icon",
        text="",
        type="figure",
        block_type=BlockType.FIGURE,
        page_number=1,
        bbox=(0.0, 0.0, 20.0, 20.0),  # 400 pt^2 — far below the default threshold
    )
    document = Document(file_name="icons.pdf", file_type="pdf", blocks=[icon])

    vlm = CountingVLM()
    DocumentEnhancer(vlm_client=vlm).enhance(document, pdf_path)

    assert vlm.calls == 0
    assert icon.vlm_description is None
