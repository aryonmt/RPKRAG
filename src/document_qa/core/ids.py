import hashlib
import uuid

# Constant namespace for our project. 
# WARNING: Changing this changes all node IDs and requires a reindex.
NAMESPACE_DOCUMENT_QA = uuid.UUID("9a8b7c6d-5e4f-3a2b-1c0d-9e8f7a6b5c4d")


def generate_document_id() -> str:
    """Generates a random UUID for a new document."""
    return str(uuid.uuid4())


def generate_conversation_id() -> str:
    """Generates a random UUID for a new conversation."""
    return str(uuid.uuid4())


def generate_message_id() -> str:
    """Generates a random UUID for a new message."""
    return str(uuid.uuid4())


def generate_node_id(document_id: str, ordinal: int, original_text: str) -> str:
    """
    Generates a deterministic UUID for a chunk node.
    
    If the same document is processed again with the same text, the node_id 
    will be identical. This helps with idempotency and re-indexing.
    """
    text_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
    name = f"{document_id}:{ordinal}:{text_hash}"
    
    return str(uuid.uuid5(NAMESPACE_DOCUMENT_QA, name))