from file_agent.parsers.html_parser import HTMLParser


def test_html_parser_extracts_readable_text(tmp_path):
    file_path = tmp_path / "page.html"
    file_path.write_text(
        "<html><body><h1>Title</h1><p>Hello HTML</p></body></html>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(file_path)

    assert document.file_name == "page.html"
    assert document.file_type == "html"
    assert len(document.blocks) == 1
    assert document.blocks[0].id == "block-1"
    assert document.blocks[0].type == "html_text"
    assert "Title" in document.blocks[0].text
    assert "Hello HTML" in document.blocks[0].text
    assert document.blocks[0].metadata == {"source_file": "page.html"}


def test_html_parser_excludes_script_and_style_content(tmp_path):
    file_path = tmp_path / "page.htm"
    file_path.write_text(
        """
        <html>
            <head>
                <style>.hidden { color: red; }</style>
                <script>console.log("secret")</script>
            </head>
            <body><p>Visible text</p></body>
        </html>
        """,
        encoding="utf-8",
    )

    document = HTMLParser().parse(file_path)
    text = document.blocks[0].text

    assert "Visible text" in text
    assert "hidden" not in text
    assert "console.log" not in text
    assert "secret" not in text
