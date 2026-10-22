import re
from pathlib import Path
from document_qa.ingestion.normalizer import normalize_for_search, tokenize, remove_stopwords, load_stopwords

def build_fts_query(question: str) -> str:
    stopwords_path = Path(__file__).parent.parent.parent.parent / "resources" / "fa_stopwords.txt"
    stopwords = load_stopwords(str(stopwords_path))
    
    normalized = normalize_for_search(question)
    tokens = tokenize(normalized)
    
    content_tokens = remove_stopwords(tokens, stopwords)
    if not content_tokens: content_tokens = tokens
    if not content_tokens: return ""
        
    seen = set()
    unique_tokens = [x for x in content_tokens if not (x in seen or seen.add(x))]
    unique_tokens = unique_tokens[:32]
    
    safe_tokens = []
    for t in unique_tokens:
        if re.match(r'^[A-Za-z]+-\d+', t):
            safe_tokens.append(f'"{t}"')
        else:
            safe_tokens.append(f'"{t.replace(chr(34), chr(34)+chr(34))}"')
    
    return " OR ".join(safe_tokens)