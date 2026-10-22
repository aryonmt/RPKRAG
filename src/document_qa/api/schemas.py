from pydantic import BaseModel, Field
from typing import Optional

# Conversation Schemas
class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=120)

class ConversationResponse(BaseModel):
    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    document_count: int = 0
    message_count: int = 0

class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


# Message & Chat Schemas
class MessageCreate(BaseModel):
    question: str

class CitationSchema(BaseModel):
    evidence_id: str
    filename: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    breadcrumb: Optional[str] = None
    quote: str
    quote_truncated: bool

class MessageResponse(BaseModel):
    message_id: str
    question: str
    standalone_question: Optional[str] = None
    status: str
    answer_text: str
    missing: list[str]
    citations: list[CitationSchema]