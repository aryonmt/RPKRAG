import re
import unicodedata

def fold_digits(text: str) -> str:
    """Map Persian and Arabic-Indic digits to ASCII. Does not otherwise change the text."""
    if not text:
        return ""
    text = text.replace("٫", ".").replace("٪", "%")
    for i in range(10):
        text = text.replace(chr(0x06F0 + i), str(i))
        text = text.replace(chr(0x0660 + i), str(i))
    return text


def normalize_for_search(text: str) -> str:
    """
    Normalizes Persian text for search, embedding, and FTS.
    Does not change the original text used for citations.
    """
    if not text:
        return ""

    # 1. NFKC normalization (converts presentation forms to base letters)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove direction-control, zero-width chars, and tatweel
    # U+200B (ZWSP), U+200E, U+200F, U+202A-U+202E, U+FEFF, U+0640 (Tatweel)
    text = re.sub(r"[\u200B\u200E\u200F\u202A-\u202E\uFEFF\u0640]", "", text)

    # 3. Map Arabic letters to Persian
    # ي (U+064A), ى (U+0649), ے (U+06D2) -> ی (U+06CC)
    # ك (U+0643), ڪ (U+06AA) -> ک (U+06A9)
    translation_map = str.maketrans("يىےكڪ", "یییکک")
    text = text.translate(translation_map)

    # 4. Map Persian and Arabic-Indic digits to Latin
    text = fold_digits(text)

    # 5. Remove diacritics (U+064B to U+065F and U+0670)
    # BUT preserve U+0654 (Ezafe marker)
    text = re.sub(r"[\u064B-\u0653\u0655-\u065F\u0670]", "", text)

    # 6. Convert ZWNJ (U+200C) to normal space
    text = text.replace("\u200C", " ")

    # 7. Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text: str) -> list[str]:
    """
    Tokenizes normalized text into words.
    Tokens are sequences of Unicode letters and digits.
    """
    normalized = normalize_for_search(text)
    return re.findall(r"\w+", normalized)

def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Removes stopwords from a list of tokens."""
    return [t for t in tokens if t not in stopwords]

def load_stopwords(file_path: str) -> set[str]:
    """Loads stopwords from a file, one word per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and not line.startswith("#")}
    except FileNotFoundError:
        return set()