from pathlib import Path

from document_qa.ingestion.normalizer import (
    fold_digits,
    load_stopwords,
    normalize_for_search,
    remove_stopwords,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fold_digits_covers_persian_and_arabic_indic():
    assert fold_digits("۱۲۳٧٨٩") == "123789"


def test_normalize_maps_arabic_letters_and_digits():
    assert normalize_for_search("مي‌شود كِتاب ۱۲۳") == "می شود کتاب 123"


def test_stopwords_file_loads():
    words = load_stopwords(str(ROOT / "resources" / "fa_stopwords.txt"))
    assert words
    assert "و" in words
    tokens = tokenize("این یک کتاب است")
    assert "کتاب" in remove_stopwords(tokens, words)
