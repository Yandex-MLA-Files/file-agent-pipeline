"""Word's automatic list numbering, replayed from numbering.xml."""

from docx.oxml.ns import qn
from lxml import etree

from file_agent.parsers.docx_numbering import DocxNumbering

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _numbering(levels: str, nums: str = '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'):
    xml = f'<w:numbering xmlns:w="{W}"><w:abstractNum w:abstractNumId="0">{levels}</w:abstractNum>{nums}</w:numbering>'
    return DocxNumbering(etree.fromstring(xml))


DECIMAL_TWO_LEVELS = """
<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>
<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1.%2."/></w:lvl>
"""


def test_multilevel_counters_restart_under_their_parent():
    numbering = _numbering(DECIMAL_TWO_LEVELS)

    markers = [
        numbering.marker("1", 0),
        numbering.marker("1", 1),
        numbering.marker("1", 1),
        numbering.marker("1", 0),
        numbering.marker("1", 1),
    ]

    assert markers == ["1.", "1.1.", "1.2.", "2.", "2.1."]


def test_letters_and_roman_formats():
    letters = _numbering(
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/>'
        '<w:lvlText w:val="%1)"/></w:lvl>'
    )
    roman = _numbering(
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="upperRoman"/>'
        '<w:lvlText w:val="%1."/></w:lvl>'
    )

    assert [letters.marker("1", 0) for _ in range(3)] == ["a)", "b)", "c)"]
    assert [roman.marker("1", 0) for _ in range(4)] == ["I.", "II.", "III.", "IV."]


def test_bullets_and_unknown_lists_produce_no_marker():
    bullets = _numbering(
        '<w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val="\uf0b7"/></w:lvl>'
    )

    assert bullets.marker("1", 0) == ""
    assert bullets.marker("99", 0) == ""  # numId that no definition covers
    assert DocxNumbering(None).marker("1", 0) == ""
    assert DocxNumbering(None).available is False


def test_start_override_of_a_list_instance():
    numbering = _numbering(
        '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>'
        '<w:lvlText w:val="%1."/></w:lvl>',
        nums='<w:num w:numId="1"><w:abstractNumId w:val="0"/>'
        '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="8"/></w:lvlOverride></w:num>',
    )

    assert [numbering.marker("1", 0) for _ in range(2)] == ["8.", "9."]


def test_malformed_numbering_is_ignored():
    broken = DocxNumbering(
        etree.fromstring(f'<w:numbering xmlns:w="{W}"><w:abstractNum/></w:numbering>')
    )

    assert broken.available is False
    assert broken.marker("1", 0) == ""
    assert qn("w:numbering")  # namespace helper still usable
