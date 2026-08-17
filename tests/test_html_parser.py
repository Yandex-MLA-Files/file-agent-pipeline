from file_agent.document import BlockType
from file_agent.parsers.html_parser import HTMLParser
from file_agent.parsers.legacy import HTMLParser as LegacyHTMLParser


def test_html_parser_emits_typed_blocks(tmp_path):
    file_path = tmp_path / "page.html"
    file_path.write_text(
        "<html><head><title>Doc title</title></head><body>"
        "<h1>Title</h1><p>Hello HTML</p>"
        "<ul><li>one</li><li>two<ul><li>nested</li></ul></li></ul>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<pre>x = 1</pre>"
        "</body></html>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(file_path)

    assert document.file_name == "page.html"
    assert document.file_type == "html"
    assert document.metadata["title"] == "Doc title"
    types = [b.block_type for b in document.blocks]
    assert types == [
        BlockType.HEADING,
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.CODE,
    ]
    assert document.blocks[0].text == "Title"
    assert document.blocks[0].metadata["hierarchy_level"] == 1
    assert document.blocks[1].text == "Hello HTML"
    assert document.blocks[2].text == "- one\n- two\n  - nested"
    assert document.blocks[3].text.splitlines() == ["| A | B |", "| --- | --- |", "| 1 | 2 |"]
    assert document.blocks[4].text == "x = 1"


def test_html_parser_excludes_script_and_style_content(tmp_path):
    file_path = tmp_path / "page.htm"
    file_path.write_text(
        """
        <html>
            <head>
                <style>.hidden { color: red; }</style>
                <script>console.log("secret")</script>
            </head>
            <body><nav>menu</nav><p>Visible text</p></body>
        </html>
        """,
        encoding="utf-8",
    )

    document = HTMLParser().parse(file_path)
    text = "\n".join(block.text for block in document.blocks)

    assert "Visible text" in text
    assert "hidden" not in text
    assert "console.log" not in text
    assert "secret" not in text
    assert "menu" not in text


def test_legacy_html_parser_returns_one_text_block(tmp_path):
    file_path = tmp_path / "page.html"
    file_path.write_text("<html><body><h1>Title</h1><p>Hello</p></body></html>", encoding="utf-8")

    document = LegacyHTMLParser().parse(file_path)

    assert len(document.blocks) == 1
    assert document.blocks[0].type == "html_text"


def test_merged_table_cells_keep_values_under_their_own_headers(tmp_path):
    """colspan/rowspan shift every following cell if they are ignored."""
    html = """
    <table>
      <tr><th>Регион</th><th colspan="2">2026</th></tr>
      <tr><th></th><th>План</th><th>Факт</th></tr>
      <tr><td rowspan="2">Юг</td><td>100</td><td>90</td></tr>
      <tr><td>200</td><td>210</td></tr>
    </table>
    """
    path = tmp_path / "report.html"
    path.write_text(html, encoding="utf-8")

    document = HTMLParser().parse(path)
    table = next(b for b in document.blocks if b.block_type == BlockType.TABLE)
    rows = [line for line in table.text.splitlines() if "|" in line]

    # The spanning header covers both of its columns ...
    assert rows[0] == "| Регион | 2026 | 2026 |"
    # ... and the row label spans down, so the second data row is not shifted
    # left into the label column.
    assert rows[-1] == "| Юг | 200 | 210 |"


def test_absurd_spans_do_not_explode_the_grid(tmp_path):
    path = tmp_path / "wide.html"
    path.write_text(
        '<table><tr><td colspan="9999">x</td></tr><tr><td>a</td><td>b</td></tr></table>',
        encoding="utf-8",
    )

    document = HTMLParser().parse(path)
    table = next(b for b in document.blocks if b.block_type == BlockType.TABLE)

    assert max(len(line.split("|")) for line in table.text.splitlines()) <= 44
