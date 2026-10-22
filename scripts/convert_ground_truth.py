import json
import re
from pathlib import Path

MD_FILE = Path("docs/samples/large-samples/11_ground_truth_qa.md")
OUT_FILE = Path("eval/ground_truth.jsonl")

def parse_md():
    if not MD_FILE.exists():
        print(f"❌ File {MD_FILE} not found.")
        return

    content = MD_FILE.read_text(encoding="utf-8")
    blocks = re.split(r'\n### (Q\d+)', content)
    
    results = []
    for i in range(1, len(blocks), 2):
        q_id = blocks[i]
        block_text = blocks[i+1]
        
        cat_match = re.search(r'\((S|M|N|C|T)\)', block_text)
        category = cat_match.group(1) if cat_match else "Unknown"
        
        q_match = re.search(r'\*\*پرسش:\*\*\s*(.*?)\n', block_text)
        question = q_match.group(1).strip() if q_match else ""
        
        a_match = re.search(r'\*\*پاسخ مورد انتظار:\*\*\s*(.*?)\n', block_text)
        expected = a_match.group(1).strip() if a_match else ""
        
        src_match = re.search(r'\*\*مرجع:\*\*\s*(.*?)\n', block_text)
        source = src_match.group(1).strip() if src_match else ""
        
        results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "expected_answer": expected,
            "source": source
        })
        
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"✅ Converted {len(results)} questions to {OUT_FILE}")

if __name__ == "__main__":
    parse_md()