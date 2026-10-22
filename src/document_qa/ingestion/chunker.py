import re
from pathlib import Path
from tokenizers import Tokenizer
from document_qa.core.config import settings
from document_qa.ingestion.normalizer import normalize_for_search
from document_qa.ingestion.headings import HeadingState
from document_qa.core.ids import generate_node_id

class LocalTokenizer:
    def __init__(self, model_dir: str):
        tokenizer_path = Path(model_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.num_special_tokens = 2

    def count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(self._tokenizer.encode(text).ids)

    def count_embedding_tokens(self, breadcrumb: str | None, body: str) -> int:
        text = normalize_for_search((breadcrumb or "") + "\n" + body)
        return self.count_tokens(text) + self.num_special_tokens

def _split_into_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?؟؛;])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def _split_oversized_text(text: str, tokenizer: LocalTokenizer) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        test_chunk = " ".join(current_chunk + [word])
        if tokenizer.count_tokens(normalize_for_search(test_chunk)) > settings.CHUNK_MAX_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap = _overlap_words(current_chunk, tokenizer)
            current_chunk = overlap + [word]
            while len(current_chunk) > 1 and tokenizer.count_tokens(normalize_for_search(" ".join(current_chunk))) > settings.CHUNK_MAX_TOKENS:
                current_chunk.pop(0)
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def _overlap_words(words: list[str], tokenizer: LocalTokenizer) -> list[str]:
    overlap = words[-20:] if len(words) > 20 else list(words)
    limit = settings.CHUNK_OVERLAP_TOKENS
    while overlap and tokenizer.count_tokens(normalize_for_search(" ".join(overlap))) > limit:
        overlap.pop(0)
    return overlap


def _locate(page_text: str, snippet: str, cursor: int) -> int:
    if not snippet:
        return max(cursor, 0)
    idx = page_text.find(snippet, max(cursor, 0))
    if idx == -1:
        return max(cursor, 0)
    return idx


def _flush_chunk(body, state, chunks, ordinal, doc_id, filename, title, page_num, tokenizer, char_start, page_text):
    body = body.strip()
    if not body: return

    if char_start == -1:
        char_start = _locate(page_text, body, 0)

    chunk = {
        "node_id": generate_node_id(doc_id, ordinal, body),
        "document_id": doc_id, "ordinal": ordinal, "filename": filename, "title": title,
        "page_start": page_num, "page_end": page_num, "breadcrumb": state.get_breadcrumb(),
        "original_text": body, "search_text": normalize_for_search(body),
        "char_start": char_start, "char_end": char_start + len(body),
        "token_count": tokenizer.count_tokens(normalize_for_search(body))
    }
    chunks.append(chunk)

def chunk_pages(pages: list[dict], document_id: str, filename: str, title: str, tokenizer: LocalTokenizer) -> list[dict]:
    chunks = []
    state = HeadingState()
    ordinal = 0

    for page in pages:
        if page["is_low_text"]: continue

        page_text = page["text"]
        page_number = page["page_number"]

        if filename.endswith(".md"):
            blocks = re.split(r'\n(?=#{1,3}\s)', page_text)
        else:
            blocks = re.split(r"\n\s*\n", page_text)

        current_body = ""
        current_char_start = 0
        cursor = 0

        for block in blocks:
            stripped_block = block.strip()
            if not stripped_block: continue

            # FIXED: Check heading without updating state
            if state.is_heading(stripped_block.split('\n')[0].strip()):
                if current_body:
                    _flush_chunk(current_body, state, chunks, ordinal, document_id, filename, title, page_number, tokenizer, current_char_start, page_text)
                    ordinal += 1
                    current_body = ""
                
                # NOW update state and start new chunk
                state.process_line(stripped_block.split('\n')[0].strip())
                current_body = stripped_block
                current_char_start = _locate(page_text, stripped_block, cursor)
                cursor = current_char_start + len(stripped_block)
                continue

            sentences = _split_into_sentences(stripped_block)
            for sentence in sentences:
                potential = (current_body + " " + sentence).strip() if current_body else sentence
                
                body_tokens = tokenizer.count_tokens(normalize_for_search(potential))
                embed_tokens = tokenizer.count_embedding_tokens(state.get_breadcrumb(), potential)

                if body_tokens > settings.CHUNK_MAX_TOKENS or embed_tokens > settings.EMBEDDING_MAX_TOKENS:
                    if current_body:
                        _flush_chunk(current_body, state, chunks, ordinal, document_id, filename, title, page_number, tokenizer, current_char_start, page_text)
                        ordinal += 1
                        cursor = max(cursor, current_char_start + len(current_body))
                    
                    words = current_body.split()
                    overlap_text = " ".join(_overlap_words(words, tokenizer))
                    
                    if tokenizer.count_tokens(normalize_for_search(sentence)) > settings.CHUNK_MAX_TOKENS:
                        hard_chunks = _split_oversized_text(sentence, tokenizer)
                        for hc in hard_chunks:
                            start = _locate(page_text, hc, cursor)
                            _flush_chunk(hc, state, chunks, ordinal, document_id, filename, title, page_number, tokenizer, start, page_text)
                            ordinal += 1
                            cursor = start + len(hc)
                        current_body = ""
                        current_char_start = -1
                    else:
                        current_body = (overlap_text + " " + sentence).strip() if overlap_text else sentence
                        too_long = (
                            tokenizer.count_tokens(normalize_for_search(current_body)) > settings.CHUNK_MAX_TOKENS
                            or tokenizer.count_embedding_tokens(state.get_breadcrumb(), current_body) > settings.EMBEDDING_MAX_TOKENS
                        )
                        if too_long:
                            parts = _split_oversized_text(current_body, tokenizer)
                            for hc in parts[:-1]:
                                start = _locate(page_text, hc, cursor)
                                _flush_chunk(hc, state, chunks, ordinal, document_id, filename, title, page_number, tokenizer, start, page_text)
                                ordinal += 1
                                cursor = start + len(hc)
                            current_body = parts[-1] if parts else ""
                        current_char_start = _locate(page_text, current_body, cursor) if current_body else cursor
                        cursor = current_char_start + len(current_body)
                else:
                    if not current_body:
                        current_char_start = _locate(page_text, sentence, cursor)
                        cursor = current_char_start + len(sentence)
                    current_body = potential

        if current_body:
            too_long = (
                tokenizer.count_tokens(normalize_for_search(current_body)) > settings.CHUNK_MAX_TOKENS
                or tokenizer.count_embedding_tokens(state.get_breadcrumb(), current_body) > settings.EMBEDDING_MAX_TOKENS
            )
            pieces = _split_oversized_text(current_body, tokenizer) if too_long else [current_body]
            for hc in pieces:
                start = _locate(page_text, hc, cursor)
                _flush_chunk(hc, state, chunks, ordinal, document_id, filename, title, page_number, tokenizer, start, page_text)
                ordinal += 1
                cursor = start + len(hc)

    return chunks