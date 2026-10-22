from document_qa.core.config import settings
from document_qa.ingestion.chunker import chunk_pages
from document_qa.ingestion.normalizer import normalize_for_search


class WordTokenizer:
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def count_embedding_tokens(self, breadcrumb: str | None, body: str) -> int:
        return self.count_tokens(normalize_for_search((breadcrumb or "") + "\n" + body)) + 2


def test_chunks_stay_within_token_caps():
    tokenizer = WordTokenizer()
    sentence = "این یک جمله تستی است که کلمات زیادی دارد. "
    pages = [{
        "page_number": 1,
        "text": f"فصل اول: مقدمه\n\n{sentence * 40}\n\nماده ۲: بخش دوم\n\n{sentence * 40}",
        "is_low_text": False,
    }]
    chunks = chunk_pages(pages, "doc", "test.pdf", "سند", tokenizer)
    assert chunks
    for chunk in chunks:
        assert chunk["token_count"] <= settings.CHUNK_MAX_TOKENS
        embed_tokens = tokenizer.count_embedding_tokens(chunk["breadcrumb"], chunk["original_text"])
        assert embed_tokens <= settings.EMBEDDING_MAX_TOKENS
        assert chunk["char_end"] >= chunk["char_start"]
