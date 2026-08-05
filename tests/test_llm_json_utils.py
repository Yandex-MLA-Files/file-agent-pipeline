import json

import pytest

from file_agent.llm.json_utils import extract_json_object


def test_extract_json_object_returns_bare_object():
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_strips_surrounding_prose():
    assert extract_json_object('Ответ: {"a": 1} спасибо') == '{"a": 1}'


def test_extract_json_object_raises_when_no_braces_found():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("no json here")
