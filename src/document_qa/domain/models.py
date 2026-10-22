from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    PARTIAL_WARNING = "partial_warning"
    FAILED = "failed"
    DELETING = "deleting"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    DELETING = "deleting"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    CONFLICT = "conflict"


class Conversation(BaseModel):
    id: str
    title: str
    status: ConversationStatus
    created_at: str
    updated_at: str


class Document(BaseModel):
    id: str
    conversation_id: str
    filename: str
    title: Optional[str] = None
    file_type: str
    file_hash: str
    status: DocumentStatus
    generation: int
    page_count: Optional[int] = None
    chunk_count: int = 0
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    cleanup_pending: bool = False
    raw_file_path: Optional[str] = None
    created_at: str
    updated_at: str


class Chunk(BaseModel):
    node_id: str
    conversation_id: str
    document_id: str
    ordinal: int
    filename: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    breadcrumb: Optional[str] = None
    original_text: str
    search_text: str
    char_start: int
    char_end: int
    token_count: int


class Citation(BaseModel):
    evidence_id: str
    filename: str
    title: str
    page_start: Optional[int]
    page_end: Optional[int]
    breadcrumb: Optional[str]
    quote: str
    quote_truncated: bool


class Message(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    standalone_question: Optional[str] = None
    status: Optional[AnswerStatus] = None
    answer_text: Optional[str] = None
    missing: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    created_at: str