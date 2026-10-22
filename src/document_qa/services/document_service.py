import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from fastapi import UploadFile
from document_qa.core.config import settings
from document_qa.storage.sqlite import get_db_connection
from document_qa.api.errors import AppError
from document_qa.retrieval.qdrant_index import delete_document_points, get_qdrant_client
from qdrant_client.http import models

def _get_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = "".join(c for c in name if ord(c) >= 32)
    return name.strip()[:120] or "document"

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def process_upload(conversation_id: str, file: UploadFile) -> dict:
    conn = get_db_connection()
    conv = conn.execute("SELECT id FROM conversations WHERE id = ? AND status = 'active'", (conversation_id,)).fetchone()
    if not conv:
        conn.close()
        raise AppError(404, "conversation_not_found", "این گفتگو پیدا نشد.")

    filename = safe_filename(file.filename or "unknown")
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        conn.close()
        raise AppError(422, "unsupported_file_type", "فقط فایل PDF، TXT و Markdown پشتیبانی می‌شود.")

    doc_id = str(uuid.uuid4())
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}{ext}"
    
    max_size_bytes = settings.MAX_FILE_MB * 1024 * 1024
    total_size = 0
    
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size_bytes:
                    raise AppError(413, "file_too_large", "حجم فایل از حد مجاز بیشتر است.")
                buffer.write(chunk)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        conn.close()
        if isinstance(e, AppError):
            raise e
        raise AppError(500, "internal_error", "خطا در ذخیره فایل.")

    if ext == ".pdf":
        with open(file_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                file_path.unlink()
                conn.close()
                raise AppError(422, "invalid_document", "فایل قابل‌خواندن نیست یا قالب آن معتبر نیست.")

    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE conversation_id = ? AND status != 'deleting'",
        (conversation_id,),
    ).fetchone()
    if existing and existing["n"] >= settings.MAX_DOCUMENTS_PER_CONVERSATION:
        file_path.unlink(missing_ok=True)
        conn.close()
        raise AppError(422, "too_many_documents", "تعداد سندهای این گفتگو از حد مجاز بیشتر است.")

    file_hash = calculate_sha256(file_path)
    dup = conn.execute(
        "SELECT * FROM documents WHERE conversation_id = ? AND file_hash = ? AND status != 'deleting'",
        (conversation_id, file_hash)
    ).fetchone()

    if dup:
        file_path.unlink()
        conn.close()
        return {"duplicate": True, "document_id": dup["id"]}

    now = _get_now()
    # Added raw_file_path to the insert query
    conn.execute("""
        INSERT INTO documents (
            id, conversation_id, filename, file_type, file_hash, status, generation, raw_file_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'processing', 1, ?, ?, ?)
    """, (doc_id, conversation_id, filename, ext.replace(".", ""), file_hash, str(file_path), now, now))
    conn.commit()
    conn.close()

    return {"duplicate": False, "document_id": doc_id, "file_path": str(file_path)}


def delete_document(conversation_id: str, document_id: str):
    conn = get_db_connection()
    doc = conn.execute("SELECT raw_file_path FROM documents WHERE id = ? AND conversation_id = ?", (document_id, conversation_id)).fetchone()
    if not doc:
        conn.close()
        raise AppError(404, "document_not_found", "این سند در این گفتگو پیدا نشد.")

    conn.execute("UPDATE documents SET status = 'deleting', updated_at = ? WHERE id = ?", (_get_now(), document_id))
    conn.commit()

    try:
        delete_document_points(document_id)
        if doc["raw_file_path"]:
            p = Path(doc["raw_file_path"])
            if p.exists():
                p.unlink()
        conn.execute("DELETE FROM chunks_fts WHERE node_id IN (SELECT node_id FROM chunks WHERE document_id = ?)", (document_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
    except Exception:
        conn.execute("UPDATE documents SET cleanup_pending = 1 WHERE id = ?", (document_id,))
        conn.commit()
        raise AppError(503, "cleanup_failed", "حذف کامل انجام نشد. دوباره تلاش کنید.")
    finally:
        conn.close()


def delete_conversation(conversation_id: str):
    conn = get_db_connection()
    conv = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND status IN ('active', 'deleting')",
        (conversation_id,),
    ).fetchone()
    if not conv:
        conn.close()
        raise AppError(404, "conversation_not_found", "این گفتگو پیدا نشد.")
        
    conn.execute("UPDATE conversations SET status = 'deleting', updated_at = ? WHERE id = ?", (_get_now(), conversation_id))
    conn.execute("UPDATE documents SET status = 'deleting' WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    
    try:
        # 1. Qdrant
        client = get_qdrant_client()
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="conversation_id", match=models.MatchValue(value=conversation_id))])
            ),
        )
        
        # 2. Files
        docs = conn.execute("SELECT raw_file_path FROM documents WHERE conversation_id = ?", (conversation_id,)).fetchall()
        for d in docs:
            if d["raw_file_path"]:
                p = Path(d["raw_file_path"])
                if p.exists():
                    p.unlink()
        
        # 3. SQLite
        conn.execute("DELETE FROM chunks_fts WHERE node_id IN (SELECT node_id FROM chunks WHERE conversation_id = ?)", (conversation_id,))
        conn.execute("DELETE FROM chunks WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM documents WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    except Exception:
        raise AppError(503, "cleanup_failed", "حذف کامل انجام نشد. دوباره تلاش کنید.")
    finally:
        conn.close()