from pathlib import Path
import pymupdf as fitz
from document_qa.core.config import settings

def fix_persian_pdf_text(text: str) -> str:
    return text.replace("صالحیت", "صلاحیت")

def is_garbage_page(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    letters = sum(1 for c in stripped if c.isalpha())
    digits = sum(1 for c in stripped if c.isdigit())
    if letters == 0:
        return True
    return digits > letters * 2 and letters < 40

def extract_pdf(file_path: Path) -> list[dict]:
    try:
        doc = fitz.open(file_path)
    except Exception:
        raise ValueError("invalid_document")

    if doc.is_encrypted:
        raise ValueError("pdf_encrypted")

    if doc.page_count > settings.MAX_PDF_PAGES:
        raise ValueError("pdf_too_many_pages")

    pages = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text", sort=True)
        text = fix_persian_pdf_text(text)
        
        low_text = len(text.strip()) < settings.LOW_TEXT_PAGE_CHARS
        garbage = is_garbage_page(text)
        
        pages.append({
            "page_number": page_idx + 1,
            "text": text,
            "is_low_text": low_text or garbage
        })

    doc.close()
    return pages

def extract_text_file(file_path: Path) -> list[dict]:
    try:
        # utf-8-sig handles Windows BOM issues
        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError:
        raise ValueError("invalid_document")
    
    return [{
        "page_number": None,
        "text": text,
        "is_low_text": len(text.strip()) < settings.LOW_TEXT_PAGE_CHARS
    }]

def extract_document(file_path: Path, file_extension: str) -> list[dict]:
    ext = file_extension.lower()
    
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                raise ValueError("invalid_document")
        return extract_pdf(file_path)
    
    elif ext in (".txt", ".md"):
        return extract_text_file(file_path)
    
    else:
        raise ValueError("unsupported_file_type")