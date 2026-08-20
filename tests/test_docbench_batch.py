from pathlib import Path

import pytest

import file_agent.docbench_batch as docbench_batch
from file_agent.chunking import Chunk
from file_agent.docbench_batch import generate_docbench_records
from file_agent.docbench_dataset import DocBenchRecord
from file_agent.document import Block, Document
from file_agent.rag_core import RAGResponse
from file_agent.retrieval import SearchResult


class FakeLLM:
    model = "fake-model"
    temperature = 0.0
    max_tokens = 100
    enable_thinking = False

    def __init__(self) -> None:
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def generate(self, prompt: str) -> str:
        raise AssertionError("The patched answer function should be used")


class FakeRetriever:
    def __init__(self) -> None:
        self.index_count = 0
        self.clear_count = 0

    def index(self, chunks: list[Chunk]) -> None:
        self.index_count += 1

    def search(self, query: str, top_k: int = 5, source_file: str | None = None):
        return []

    def clear(self) -> None:
        self.clear_count += 1


def _records(tmp_path: Path) -> tuple[DocBenchRecord, DocBenchRecord]:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    qa_path = tmp_path / "0_qa.jsonl"
    qa_path.write_text("{}\n", encoding="utf-8")
    base = {
        "folder_id": 0,
        "reference_answer": "gold",
        "question_type": "text-only",
        "evidence": "evidence",
        "domain": "academia",
        "pdf_path": pdf_path,
        "qa_path": qa_path,
    }
    return (
        DocBenchRecord(
            id="docbench-000-q0001",
            question_index=0,
            question="Question one?",
            **base,
        ),
        DocBenchRecord(
            id="docbench-000-q0002",
            question_index=1,
            question="Question two?",
            **base,
        ),
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, llm: FakeLLM, fail_question: str = ""):
    loaders: list[Path] = []

    def load_document(path: Path, vlm_client):
        loaders.append(path)
        return Document(
            file_name=path.name,
            file_type="pdf",
            blocks=[Block(id="b1", text="document evidence", type="text")],
            metadata={"total_pages": 2},
        )

    def ingest_documents(documents, retriever, max_chars, overlap):
        chunk = Chunk(
            id="c1",
            text="document evidence",
            metadata=dict(documents[0].blocks[0].metadata),
        )
        retriever.index([chunk])
        return [chunk]

    def answer_indexed_documents(*, question, **kwargs):
        if question == fail_question:
            raise TimeoutError("simulated timeout")
        llm.request_count += 1
        llm.prompt_tokens += 10
        llm.completion_tokens += 2
        llm.total_tokens += 12
        source = SearchResult(
            chunk=Chunk(
                id="c1",
                text="document evidence",
                metadata={
                    "dataset_doc_id": "0/document.pdf",
                    "source_file": "document.pdf",
                    "context": "document evidence",
                },
            ),
            score=0.9,
        )
        return RAGResponse(
            answer=f"Answer to {question}",
            sources=[source],
            documents_count=1,
            chunks_count=1,
            search_queries=[question],
            tool_calls=[{"name": "search_documents"}],
        )

    monkeypatch.setattr(docbench_batch, "ingest_documents", ingest_documents)
    monkeypatch.setattr(docbench_batch, "answer_indexed_documents", answer_indexed_documents)
    return load_document, loaders


def test_batch_parses_and_indexes_once_then_resumes_without_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    llm = FakeLLM()
    loader, loaded_paths = _patch_pipeline(monkeypatch, llm)
    retrievers: list[FakeRetriever] = []

    def retriever_factory() -> FakeRetriever:
        retriever = FakeRetriever()
        retrievers.append(retriever)
        return retriever

    first = generate_docbench_records(
        records,
        llm_client=llm,
        output_dir=tmp_path / "run",
        rag_mode="standard",
        document_loader=loader,
        retriever_factory=retriever_factory,
    )

    assert first.processed_count == 2
    assert first.resumed_count == 0
    assert not first.failures
    assert len(loaded_paths) == 1
    assert len(retrievers) == 1
    assert retrievers[0].index_count == 1
    assert retrievers[0].clear_count == 1
    assert llm.request_count == 2
    assert first.records[0].contexts[0].document_id == "0/document.pdf"

    def forbidden_loader(path: Path, vlm_client):
        raise AssertionError("A fully resumed document must not be parsed")

    second = generate_docbench_records(
        records,
        llm_client=llm,
        output_dir=tmp_path / "run",
        rag_mode="standard",
        resume=True,
        document_loader=forbidden_loader,
        retriever_factory=lambda: pytest.fail("A fully resumed document must not be indexed"),
    )

    assert second.processed_count == 0
    assert second.resumed_count == 2
    assert [record.id for record in second.records] == [record.id for record in records]
    assert llm.request_count == 2


def test_batch_checkpoints_success_and_continues_after_one_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    llm = FakeLLM()
    loader, _ = _patch_pipeline(monkeypatch, llm, fail_question="Question one?")

    result = generate_docbench_records(
        records,
        llm_client=llm,
        output_dir=tmp_path / "run",
        rag_mode="standard",
        document_loader=loader,
        retriever_factory=FakeRetriever,
    )

    assert result.processed_count == 1
    assert [record.id for record in result.records] == ["docbench-000-q0002"]
    assert [failure.id for failure in result.failures] == ["docbench-000-q0001"]
    assert (tmp_path / "run" / "checkpoints" / "docbench-000-q0002.json").exists()
    assert (tmp_path / "run" / "failures" / "docbench-000-q0001.json").exists()


def test_batch_rejects_existing_checkpoints_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    llm = FakeLLM()
    loader, _ = _patch_pipeline(monkeypatch, llm)
    arguments = {
        "llm_client": llm,
        "output_dir": tmp_path / "run",
        "rag_mode": "standard",
        "document_loader": loader,
        "retriever_factory": FakeRetriever,
    }
    generate_docbench_records(records, **arguments)

    with pytest.raises(FileExistsError, match="--resume"):
        generate_docbench_records(records, **arguments)
