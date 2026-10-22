from fastapi import APIRouter, UploadFile, File, BackgroundTasks, status
from typing import List
from document_qa.services.document_service import process_upload, delete_document
from document_qa.storage.sqlite import get_db_connection
from document_qa.ingestion.pipeline import process_document
from document_qa.api.errors import AppError
from pathlib import Path
import traceback

router = APIRouter()

def run_ingest_background(file_path: str, document_id: str, conversation_id: str, filename: str):
    try:
        process_document(
            file_path=Path(file_path),
            document_id=document_id,
            conversation_id=conversation_id,
            filename=filename,
            title=filename,
            generation=1
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Background Ingest Failed: {e}")

        conn = get_db_connection()
        try:
            row = conn.execute("SELECT status FROM documents WHERE id = ?", (document_id,)).fetchone()
            if row and row["status"] == "deleting":
                return
            conn.execute("""
                UPDATE documents
                SET status = 'failed', error_code = 'ingest_failed', error_message = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE id = ? AND status = 'processing'
            """, (str(e), document_id))
            conn.commit()
        finally:
            conn.close()

@router.post("/conversations/{conversation_id}/documents", status_code=status.HTTP_202_ACCEPTED)
def upload_documents(
    conversation_id: str, 
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    results = []
    for file in files:
        result = process_upload(conversation_id, file)
        
        if not result["duplicate"]:
            background_tasks.add_task(
                run_ingest_background,
                file_path=result["file_path"],
                document_id=result["document_id"],
                conversation_id=conversation_id,
                filename=file.filename
            )
            
        results.append({
            "duplicate": result["duplicate"], 
            "document": {"id": result["document_id"]}
        })

    return {"documents": results}

@router.get("/conversations/{conversation_id}/documents")
def list_documents(conversation_id: str):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, filename, status, chunk_count, error_code FROM documents WHERE conversation_id = ? AND status != 'deleting'",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return {"documents": [dict(r) for r in rows]}

@router.get("/conversations/{conversation_id}/documents/{document_id}")
def get_document(conversation_id: str, document_id: str):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, filename, status, chunk_count, error_code, error_message FROM documents WHERE id = ? AND conversation_id = ?",
        (document_id, conversation_id)
    ).fetchone()
    conn.close()
    if not row:
        raise AppError(404, "document_not_found", "این سند در این گفتگو پیدا نشد.")
    return dict(row)

@router.delete("/conversations/{conversation_id}/documents/{document_id}")
def remove_document(conversation_id: str, document_id: str):
    delete_document(conversation_id, document_id)
    return {"deleted": True}