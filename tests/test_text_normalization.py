import pytest

from file_agent.text_normalization import fold_yo, normalize_for_fts


def test_yo_is_folded_in_both_cases():
    assert fold_yo("Ещё учёт Ёлка") == "Еще учет Елка"


def test_normalization_lowercases_folds_and_keeps_numbers():
    text = "Выручка выросла до 12,5 млрд ₽ в 2026 году (учёт по МСФО)."
    assert normalize_for_fts(text, lemmatize=False) == (
        "выручка выросла до 12,5 млрд в 2026 году учет по мсфо"
    )


def test_hyphenated_and_latin_tokens_survive():
    text = "A/B-тест на онлайн-курсах и don't"
    assert normalize_for_fts(text, lemmatize=False) == "a b-тест на онлайн-курсах и don't"


def test_lemmatization_maps_inflected_forms_to_one_head_word():
    pytest.importorskip("pymorphy3")
    assert normalize_for_fts("людям людей человека", lemmatize=True) == "человек человек человек"
    assert normalize_for_fts("компании компаниями", lemmatize=True) == "компания компания"


def test_lemmatization_leaves_latin_and_numbers_alone():
    pytest.importorskip("pymorphy3")
    assert normalize_for_fts("MongoDB Compass 2026", lemmatize=True) == "mongodb compass 2026"


def test_empty_text_normalizes_to_empty_string():
    assert normalize_for_fts("") == ""
    assert normalize_for_fts("   \n ") == ""
