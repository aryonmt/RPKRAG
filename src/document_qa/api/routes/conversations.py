import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status
from pydantic import BaseModel
from document_qa.api.schemas import ConversationCreate, ConversationResponse, ConversationListResponse
from document_qa.api.errors import AppError
from document_qa.storage.sqlite import get_db_connection
from document_qa.services.document_service import delete_conversation

router = APIRouter()

def _get_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

class ConversationUpdate(BaseModel):
    title: str

@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: ConversationCreate):
    conn = get_db_connection()
    try:
        conv_id = str(uuid.uuid4())
        title = payload.title.strip() if payload.title else ""
        if not title:
            title = "گفتگوی جدید"
            
        now = _get_now()
        
        conn.execute(
            "INSERT INTO conversations (id, title, status, created_at, updated_at) VALUES (?, ?, 'active', ?, ?)",
            (conv_id, title, now, now)
        )
        conn.commit()
        
        return ConversationResponse(
            id=conv_id, title=title, status="active", created_at=now, updated_at=now, document_count=0, message_count=0
        )
    finally:
        conn.close()

@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations():
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT c.*, 
                   (SELECT COUNT(*) FROM documents d WHERE d.conversation_id = c.id AND d.status != 'deleting') as doc_count,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count
            FROM conversations c
            WHERE c.status = 'active'
            ORDER BY c.updated_at DESC
            LIMIT 200
        """).fetchall()
        
        convs = []
        for row in rows:
            convs.append(ConversationResponse(
                id=row["id"], title=row["title"], status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
                document_count=row["doc_count"], message_count=row["msg_count"]
            ))
        return ConversationListResponse(conversations=convs)
    finally:
        conn.close()

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str):
    conn = get_db_connection()
    try:
        row = conn.execute("""
            SELECT c.*, 
                   (SELECT COUNT(*) FROM documents d WHERE d.conversation_id = c.id AND d.status != 'deleting') as doc_count,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count
            FROM conversations c
            WHERE c.id = ? AND c.status = 'active'
        """, (conversation_id,)).fetchone()
        
        if not row:
            raise AppError(404, "conversation_not_found", "این گفتگو پیدا نشد.")
            
        return ConversationResponse(
            id=row["id"], title=row["title"], status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
            document_count=row["doc_count"], message_count=row["msg_count"]
        )
    finally:
        conn.close()

@router.patch("/conversations/{conversation_id}")
def update_conversation(conversation_id: str, payload: ConversationUpdate):
    title = payload.title.strip()
    if not title or len(title) > 120:
        raise AppError(422, "title_invalid", "عنوان باید بین ۱ تا ۱۲۰ نویسه باشد.")
        
    conn = get_db_connection()
    try:
        conv = conn.execute("SELECT id FROM conversations WHERE id = ? AND status = 'active'", (conversation_id,)).fetchone()
        if not conv:
            raise AppError(404, "conversation_not_found", "این گفتگو پیدا نشد.")
            
        now = _get_now()
        conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, conversation_id))
        conn.commit()
        return {"id": conversation_id, "title": title}
    finally:
        conn.close()

@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    delete_conversation(conversation_id)
    return {"deleted": True}