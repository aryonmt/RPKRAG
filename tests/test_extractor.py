from pathlib import Path

from document_qa.ingestion.extractor import extract_document, fix_persian_pdf_text


def test_known_pdf_typo_is_corrected_without_rewriting_other_words():
    assert fix_persian_pdf_text("صالحیت مولکول") == "صلاحیت مولکول"


def test_txt_extraction(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("سلام دنیا. این یک متن آزمایشی است که بیشتر از چهل نویسه دارد و باید استخراج شود.", encoding="utf-8")
    pages = extract_document(path, ".txt")
    assert len(pages) == 1
    assert pages[0]["page_number"] is None
    assert "سلام دنیا" in pages[0]["text"]
    assert pages[0]["is_low_text"] is False
