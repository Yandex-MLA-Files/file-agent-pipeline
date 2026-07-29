from file_agent.chunking import chunk_document
from file_agent.document import Block, BlockType, Document


class WordTokenizer:
    """Deterministic stand-in for a HuggingFace tokenizer (1 word == 1 token)."""

    model_max_length = 20

    def encode(self, text, **kwargs):
        return text.split()


def _document(*texts, block_type=BlockType.TEXT):
    blocks = [
        Block(id=f"b{i}", text=text, type=block_type.value, block_type=block_type)
        for i, text in enumerate(texts)
    ]
    return Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)


def test_token_budget_limits_chunks_by_tokens_not_characters():
    # 60 words: far below a 1000-character budget, but 3x the 20-token window.
    document = _document(" ".join(f"word{i}" for i in range(60)))

    chunks = chunk_document(document, max_chars=1000, overlap=100, tokenizer=WordTokenizer())

    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= WordTokenizer.model_max_length for chunk in chunks)


def test_explicit_max_tokens_overrides_tokenizer_limit():
    document = _document(" ".join(f"word{i}" for i in range(40)))

    chunks = chunk_document(
        document, max_chars=1000, overlap=100, max_tokens=10, tokenizer=WordTokenizer()
    )

    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)


def test_oversized_text_is_split_on_sentence_boundaries():
    sentences = [f"This is sentence number {i} of the document." for i in range(12)]
    document = _document(" ".join(sentences))

    chunks = chunk_document(document, max_chars=120, overlap=20)

    assert len(chunks) > 1
    # No chunk may start or end mid-sentence: every piece keeps whole sentences.
    for chunk in chunks:
        assert chunk.text.strip().endswith(".")
        assert chunk.text.strip()[0].isupper()


def test_words_are_never_cut_in_half():
    document = _document(" ".join(f"supercalifragilistic{i}" for i in range(30)))

    chunks = chunk_document(document, max_chars=100, overlap=10)

    rejoined = " ".join(chunks[0].text.split())
    for word in rejoined.split():
        assert word.startswith("supercalifragilistic")


def test_no_chunk_ever_exceeds_the_token_budget():
    # Adversarial mix: a heading (breadcrumb), a very wide table row, dense
    # formula-like text and one huge word — every path must respect the budget.
    wide_row = "| " + " | ".join(f"column-value-{i}" for i in range(40)) + " |"
    table = "| a | b |\n| - | - |\n" + wide_row + "\n" + wide_row
    blocks = [
        Block(id="h1", text="Results", type="heading", block_type=BlockType.HEADING),
        Block(id="t1", text=table, type="table", block_type=BlockType.TABLE),
        Block(id="f1", text="x=" + "+".join(f"a{i}" for i in range(200)), type="text"),
        Block(id="w1", text="w" * 4000, type="text"),
        Block(id="p1", text=" ".join(f"word{i}" for i in range(200)), type="text"),
    ]
    document = Document(file_name="doc.pdf", file_type="pdf", blocks=blocks)

    tokenizer = WordTokenizer()
    chunks = chunk_document(document, max_chars=1000, overlap=100, tokenizer=tokenizer)

    assert chunks
    for chunk in chunks:
        assert len(tokenizer.encode(chunk.text)) <= WordTokenizer.model_max_length, chunk.text[:80]


def test_character_mode_is_used_without_tokenizer():
    document = _document("abcdefghij")

    chunks = chunk_document(document, max_chars=4, overlap=1)

    # Legacy character windows stay exact when no tokenizer is supplied.
    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij", "j"]
