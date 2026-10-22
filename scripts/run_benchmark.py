import sys
import time
import uuid
import subprocess
from pathlib import Path

from document_qa.storage.sqlite import get_db_connection
from document_qa.ingestion.pipeline import process_document
from document_qa.services.document_service import calculate_sha256

def setup_and_run(only_failing=False, skip_upload=False):
    print("🔧 Setting up benchmark environment...")
    
    conv_id = str(uuid.uuid4())
    
    if not skip_upload:
        conn = get_db_connection()
        conn.execute("INSERT INTO conversations (id, title, status, created_at, updated_at) VALUES (?, 'Benchmark Run', 'active', 'now', 'now')", (conv_id,))
        conn.commit()
        conn.close()
        print(f"✅ Created conversation: {conv_id}")

        sample_dir = Path("docs/samples/large-samples")
        if not sample_dir.exists():
            print(f"❌ Directory {sample_dir} not found!")
            return

        files = []
        for f in sample_dir.glob("*.md"):
            if not f.name.startswith("11_"):
                files.append(f)
        for f in sample_dir.glob("*.txt"):
            files.append(f)
            
        files.sort(key=lambda x: x.name)
        print(f"📄 Found {len(files)} documents to ingest...")
        
        for f in files:
            doc_id = str(uuid.uuid4())
            conn = get_db_connection()
            now = "2025-01-01T00:00:00Z"
            conn.execute("""
                INSERT INTO documents (id, conversation_id, filename, file_type, file_hash, status, generation, raw_file_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'processing', 1, ?, ?, ?)
            """, (doc_id, conv_id, f.name, f.suffix, calculate_sha256(f), str(f), now, now))
            conn.commit()
            conn.close()
            
            try:
                print(f"  Ingesting {f.name}...")
                process_document(
                    file_path=f, document_id=doc_id, conversation_id=conv_id,
                    filename=f.name, title=f.name, generation=1
                )
            except Exception as e:
                print(f"  ❌ Failed to ingest {f.name}: {e}")
    else:
        # Find latest conversation with ingested docs
        conn = get_db_connection()
        row = conn.execute("SELECT id FROM conversations ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        if not row:
            print("❌ No conversation found. Please run without --skip-upload first.")
            return
        conv_id = row["id"]
        print(f"📂 Using existing conversation: {conv_id}")

    print("✅ Setup complete.")
    
    print("\n" + "="*40)
    print("🚀 Starting Evaluation Runner...")
    print("="*40 + "\n")
    
    cmd = [sys.executable, "eval/runner.py", conv_id]
    if only_failing:
        cmd.append("--failing")
        
    subprocess.run(cmd)

if __name__ == "__main__":
    only_failing = "--failing" in sys.argv
    skip_upload = "--skip-upload" in sys.argv
    setup_and_run(only_failing, skip_upload)