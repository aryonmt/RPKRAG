import re
from document_qa.ingestion.normalizer import normalize_for_search, tokenize

CHAPTER_PATTERN = re.compile(r"^(فصل|بخش)\s+([\d۰-۹٠-٩]+|اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)")
ARTICLE_PATTERN = re.compile(r"^ماده\s*[\(]?\s*([\d۰-۹٠-٩]+)\s*[\)]?")
NOTE_PATTERN = re.compile(r"^تبصره\s*(\d+|:|$)")
CLAUSE_PATTERN = re.compile(r"^بند\s+([\d۰-۹٠-٩]+|[آ-ی]|الف|ب|ج|د)")
MARKDOWN_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")

class HeadingState:
    def __init__(self):
        self.chapter = None
        self.article = None
        self.note = None
        self.clause = None

    def is_heading(self, line: str) -> bool:
        """Checks if a line is a heading WITHOUT updating the state."""
        line = line.strip()
        if not line or len(line) > 120:
            return False
        if MARKDOWN_PATTERN.match(line): return True
        if CHAPTER_PATTERN.match(line): return True
        if ARTICLE_PATTERN.match(line): return True
        if NOTE_PATTERN.match(line): return True
        if CLAUSE_PATTERN.match(line): return True
        return False

    def process_line(self, line: str) -> tuple[bool, str | None]:
        line = line.strip()
        if not line or len(line) > 120:
            return False, None

        md_match = MARKDOWN_PATTERN.match(line)
        if md_match:
            level = len(md_match.group(1))
            text = md_match.group(2).strip()
            if level == 1:
                self.chapter = text
                self.article = self.note = self.clause = None
            elif level == 2:
                self.article = text
                self.note = self.clause = None
            elif level == 3:
                self.note = text
                self.clause = None
            else:
                self.clause = text
            return True, self.get_breadcrumb()

        if CHAPTER_PATTERN.match(line):
            self.chapter = line
            self.article = self.note = self.clause = None
            return True, self.get_breadcrumb()

        if ARTICLE_PATTERN.match(line):
            self.article = line
            self.note = self.clause = None
            return True, self.get_breadcrumb()

        if NOTE_PATTERN.match(line):
            self.note = line
            self.clause = None
            return True, self.get_breadcrumb()

        if CLAUSE_PATTERN.match(line):
            self.clause = line
            return True, self.get_breadcrumb()

        return False, None

    def get_breadcrumb(self) -> str | None:
        parts = []
        if self.chapter: parts.append(self.chapter)
        if self.article: parts.append(self.article)
        if self.note: parts.append(self.note)
        if self.clause: parts.append(self.clause)
        if not parts: return None
        breadcrumb = " > ".join(parts)
        tokens = tokenize(breadcrumb)
        if len(tokens) > 20:
            return " ".join(tokens[:20]) + "..."
        return breadcrumb