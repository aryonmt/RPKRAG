from document_qa.ingestion.cleaner import clean_pages
from document_qa.ingestion.headings import HeadingState


def test_cleaner_drops_repeated_header_and_keeps_body():
    pages = [
        {"page_number": i, "text": f"شرکت دانش‌بنیان\n\nماده {i}: متن صفحه\n\nصفحه {i}", "is_low_text": False}
        for i in (1, 2, 3)
    ]
    cleaned, warnings = clean_pages(pages)
    assert warnings
    assert all("شرکت دانش‌بنیان" not in page["text"] for page in cleaned)
    assert "ماده 1" in cleaned[0]["text"] or "ماده ۱" in cleaned[0]["text"]


def test_heading_breadcrumb_advances():
    state = HeadingState()
    assert state.process_line("فصل اول: کلیات")[0] is True
    assert state.process_line("ماده ۱: هدف")[0] is True
    breadcrumb = state.get_breadcrumb()
    assert breadcrumb is not None
    assert "فصل اول" in breadcrumb
    assert "ماده" in breadcrumb
