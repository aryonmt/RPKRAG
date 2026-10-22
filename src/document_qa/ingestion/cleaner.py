import re
from collections import Counter

def clean_pages(pages: list[dict]) -> tuple[list[dict], list[str]]:
    warnings = []
    
    top_lines_counter = Counter()
    bottom_lines_counter = Counter()
    
    for page in pages:
        lines = [line.strip() for line in page["text"].split("\n") if line.strip()]
        if len(lines) >= 3:
            top_lines_counter.update(lines[:3])
            bottom_lines_counter.update(lines[-3:])
            
    total_pages = len(pages)
    threshold = max(3, total_pages // 2)
    
    repeated_top = {line for line, count in top_lines_counter.items() if count >= threshold}
    repeated_bottom = {line for line, count in bottom_lines_counter.items() if count >= threshold}
    repeated_lines = repeated_top | repeated_bottom

    number_pattern = re.compile(r"^\s*(صفحه|از)?\s*[\d۰-۹٠-٩]+\s*(از)?\s*[\d۰-۹٠-٩]*\s*$")

    cleaned_pages = []
    total_removed = 0

    for page in pages:
        lines = page["text"].split("\n")
        new_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
                
            is_top_or_bottom = i < 3 or i >= len(lines) - 3
            
            if is_top_or_bottom:
                if stripped in repeated_lines:
                    total_removed += 1
                    continue
                if number_pattern.match(stripped):
                    total_removed += 1
                    continue
            
            new_lines.append(line)
            
        # --- NEW LOGIC: Merge broken lines ---
        merged_lines = []
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if not stripped:
                merged_lines.append(line)
                continue
                
            if merged_lines and merged_lines[-1].strip():
                prev_stripped = merged_lines[-1].strip()
                
                # If previous line doesn't end with a sentence terminator or colon
                # AND current line doesn't look like a new heading/bullet
                if not re.search(r'[.؟!;:\)]$', prev_stripped) and not re.match(r'^(#|[-*•]|فصل|بخش|ماده|تبصره|بند)', stripped):
                    merged_lines[-1] = merged_lines[-1].rstrip() + " " + stripped
                    continue
                    
            merged_lines.append(line)
        # -------------------------------------

        text = "\n".join(merged_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        
        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": text,
            "is_low_text": bool(page.get("is_low_text")) or len(text) < 40
        })

    if total_removed > 0:
        warnings.append(f"{total_removed} خط تکراری (هدر/فوتر/شماره صفحه) حذف شد.")

    return cleaned_pages, warnings