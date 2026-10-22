import time

import pytest

pytestmark = pytest.mark.live
import httpx
from pathlib import Path

API_URL = "http://localhost:8000/api/v1"

def test_upload():
    print("--- Testing Upload API ---")
    
    # 1. Create a conversation
    res = httpx.post(f"{API_URL}/conversations", json={"title": "تست آپلود"})
    res.raise_for_status()
    conv_id = res.json()["id"]
    print(f"Created conversation: {conv_id}")

    # 2. Upload the file
    pdf_path = Path("docs/samples/sample.pdf")
    if not pdf_path.exists():
        pytest.skip("docs/samples/sample.pdf not found")

    print("Uploading file...")
    with open(pdf_path, "rb") as f:
        files = [("files", ("sample.pdf", f, "application/pdf"))]
        res = httpx.post(f"{API_URL}/conversations/{conv_id}/documents", files=files)

    res.raise_for_status()
    data = res.json()
    doc_id = data["documents"][0]["document"]["id"]
    print(f"Upload successful. Document ID: {doc_id}")

    # 3. Poll for status
    print("\nPolling document status (this will trigger background processing)...")
    while True:
        res = httpx.get(f"{API_URL}/conversations/{conv_id}/documents/{doc_id}")
        res.raise_for_status()
        doc = res.json()
        status = doc["status"]
        print(f"Status: {status} | Chunks: {doc['chunk_count']}")
        
        if status in ("indexed", "partial_warning", "failed"):
            if status == "failed":
                pytest.fail(doc.get("error_message") or "ingest failed")
            break
            
        time.sleep(2)

    print("\n✅ Upload API Test Completed.")

if __name__ == "__main__":
    test_upload()