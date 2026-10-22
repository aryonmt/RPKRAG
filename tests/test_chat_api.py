import httpx

import pytest

pytestmark = pytest.mark.live
import time
from pathlib import Path

API_URL = "http://localhost:8000/api/v1"

def test_chat_e2e():
    print("--- Testing End-to-End Chat API ---")
    
    # 1. Create a fresh conversation
    res = httpx.post(f"{API_URL}/conversations", json={"title": "تست E2E چت"})
    res.raise_for_status()
    conv_id = res.json()["id"]
    print(f"Created conversation: {conv_id}")

    # 2. Upload the PDF
    pdf_path = Path("docs/samples/sample.pdf")
    if not pdf_path.exists():
        pytest.skip("docs/samples/sample.pdf not found")

    print("Uploading file...")
    with open(pdf_path, "rb") as f:
        files = [("files", ("sample.pdf", f, "application/pdf"))]
        res = httpx.post(f"{API_URL}/conversations/{conv_id}/documents", files=files)

    res.raise_for_status()
    doc_id = res.json()["documents"][0]["document"]["id"]

    # 3. Poll for processing status
    print("Waiting for document to be processed...")
    while True:
        res = httpx.get(f"{API_URL}/conversations/{conv_id}/documents/{doc_id}")
        res.raise_for_status()
        doc = res.json()
        if doc["status"] in ("indexed", "partial_warning", "failed"):
            if doc["status"] == "failed":
                pytest.fail(doc.get("error_message") or "ingest failed")
            break
        time.sleep(2)
        
    print("Document processed. Asking question...")

    # 4. Ask the question (with a longer timeout of 150 seconds)
    question = "تمرین شماره ۱ در مورد چیست؟"
    print(f"Asking: {question}")
    
    res = httpx.post(
        f"{API_URL}/conversations/{conv_id}/messages", 
        json={"question": question},
        timeout=150.0  # <--- این خط اضافه شده است
    )
    
    if res.status_code != 200:
        pytest.fail(f"{res.status_code} {res.text}")
        
    data = res.json()
    print(f"\n--- Answer ({data['status']}) ---")
    print(data["answer_text"])
    
    if data["citations"]:
        print("\n--- Citations ---")
        for c in data["citations"]:
            page_info = f"Page {c['page_start']}" if c['page_start'] else "Text File"
            print(f"[{c['evidence_id']}] {c['title']} ({page_info})")
            print(f"Quote: {c['quote']}")
            
    print("\n✅ E2E Chat API Test Completed.")

if __name__ == "__main__":
    test_chat_e2e()