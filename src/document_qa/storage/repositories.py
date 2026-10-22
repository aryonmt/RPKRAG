import json
from sqlite3 import Connection
from document_qa.domain.models import DocumentStatus

def save_document_chunks(
    conn: Connection,
    document_id: str,
    chunks: list[dict],
    page_count: int,
    warnings: list[str],
    expected_generation: int
) -> bool:
    """
    Saves chunks to SQLite and updates the document status.
    Returns True if successful, False if the generation or status check fails.
    """
    cursor = conn.cursor()
    
    # 1. Check if the document is still in 'processing' status and generation matches
    cursor.execute(
        "SELECT status, generation FROM documents WHERE id = ?",
        (document_id,)
    )
    row = cursor.fetchone()
    if not row or row["status"] != "processing" or row["generation"] != expected_generation:
        return False

    # 2. Insert chunks and FTS data
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (
                node_id, conversation_id, document_id, ordinal, filename, title,
                page_start, page_end, breadcrumb, original_text, search_text,
                char_start, char_end, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["node_id"], chunk["conversation_id"], chunk["document_id"],
            chunk["ordinal"], chunk["filename"], chunk["title"],
            chunk["page_start"], chunk["page_end"], chunk["breadcrumb"],
            chunk["original_text"], chunk["search_text"],
            chunk["char_start"], chunk["char_end"], chunk["token_count"]
        ))
        
        cursor.execute("""
            INSERT INTO chunks_fts (node_id, body_search, breadcrumb_search, title_search)
            VALUES (?, ?, ?, ?)
        """, (
            chunk["node_id"], 
            chunk["search_text"], 
            chunk["breadcrumb"] or "", 
            chunk["title"]
        ))

    # 3. Determine final status based on warnings
    warnings_json = [{"code": "warning", "message": w} for w in warnings]
    final_status = DocumentStatus.PARTIAL_WARNING.value if warnings else DocumentStatus.INDEXED.value

    # 4. Update document status
    cursor.execute("""
        UPDATE documents 
        SET status = ?, page_count = ?, chunk_count = ?, warnings = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        WHERE id = ?
    """, (
        final_status, page_count, len(chunks), json.dumps(warnings_json, ensure_ascii=False), document_id
    ))

    conn.commit()
    return True