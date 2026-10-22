from fastapi import APIRouter, Query
from document_qa.api.schemas import MessageCreate
from document_qa.services.qa_service import ask_question
from document_qa.storage.sqlite import get_db_connection
from document_qa.api.errors import AppError
from document_qa.core.config import settings
import json

router = APIRouter()

@router.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, payload: MessageCreate, trace: bool = Query(False)):
    question = payload.question.strip()
    if not question:
        raise AppError(422, "empty_question", "سؤال نمی‌تواند خالی باشد.")
    if len(question) > settings.MAX_QUESTION_CHARS:
        raise AppError(422, "question_too_long", "سؤال از حد مجاز بلندتر است.")

    result = ask_question(conversation_id, question, return_trace=trace)
    
    # If trace is true, we return the full dictionary. Otherwise, just the result.
    return result

@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str):
    conn = get_db_connection()
    conv = conn.execute("SELECT id FROM conversations WHERE id = ? AND status = 'active'", (conversation_id,)).fetchone()
    if not conv:
        conn.close()
        raise AppError(404, "conversation_not_found", "این گفتگو پیدا نشد.")
        
    rows = conn.execute(
        "SELECT id, role, content, status, answer_text, missing, citations, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        messages.append({
            "id": row["id"], "role": row["role"], "content": row["content"],
            "status": row["status"], "answer_text": row["answer_text"],
            "missing": json.loads(row["missing"]) if row["missing"] else [],
            "citations": json.loads(row["citations"]) if row["citations"] else [],
            "created_at": row["created_at"]
        })
    return {"messages": messages}