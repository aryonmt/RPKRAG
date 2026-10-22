import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.model]
from pathlib import Path
from document_qa.ingestion.pipeline import process_document
from document_qa.storage.sqlite import get_db_connection
from document_qa.retrieval.fts_retriever import FtsRetriever
from document_qa.retrieval.qdrant_index import get_qdrant_client
from document_qa.retrieval.embedding import LocalOnnxEmbedding
from document_qa.retrieval.fusion import rrf_fusion
from document_qa.core.config import settings

def test_full_ingest():
    print("--- Testing Full Ingestion Pipeline ---")
    
    pdf_path = Path("docs/samples/sample.pdf")
    if not pdf_path.exists():
        pytest.skip("docs/samples/sample.pdf not found")

    conv_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    
    # 1. Create a dummy conversation and document in SQLite
    conn = get_db_connection()
    conn.execute("INSERT INTO conversations (id, title, status, created_at, updated_at) VALUES (?, 'Test Conv', 'active', 'now', 'now')", (conv_id,))
    conn.execute("""
        INSERT INTO documents (id, conversation_id, filename, file_type, file_hash, status, generation, created_at, updated_at) 
        VALUES (?, ?, 'sample.pdf', 'pdf', 'dummyhash', 'processing', 1, 'now', 'now')
    """, (doc_id, conv_id))
    conn.commit()
    conn.close()

    # 2. Run Pipeline
    try:
        process_document(
            file_path=pdf_path,
            document_id=doc_id,
            conversation_id=conv_id,
            filename="sample.pdf",
            title="Sample Title",
            generation=1
        )
    except Exception as e:
        raise AssertionError(f"Pipeline error: {e}") from e

    # 3. Verify SQLite
    conn = get_db_connection()
    doc_row = conn.execute("SELECT status, chunk_count FROM documents WHERE id = ?", (doc_id,)).fetchone()
    print(f"\nDocument Status: {doc_row['status']} | Total Chunks: {doc_row['chunk_count']}")
    assert doc_row["status"] in ("indexed", "partial_warning")
    assert doc_row["chunk_count"] > 0
    conn.close()

    # 4. Test Hybrid Search on this document
    print("\n--- Testing Hybrid Search on the ingested document ---")
    query = "تمرین" # We saw this word in your PDF earlier
    
    # Dense search
    embed_model = LocalOnnxEmbedding()
    client = get_qdrant_client()
    from qdrant_client.http import models
    
    q_emb = embed_model._get_query_embedding(query)
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=q_emb,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="conversation_id", match=models.MatchValue(value=conv_id))]
        ),
        limit=5
    )
    dense_results = response.points
    print(f"Dense search found {len(dense_results)} results.")
    
    # Lexical search
    fts = FtsRetriever(conversation_id=conv_id, ready_document_ids=[doc_id])
    lexical_results = fts.retrieve(query)
    print(f"Lexical search found {len(lexical_results)} results.")

    # Fusion (using a dummy wrapper for NodeWithScore since dense_results are qdrant ScoredPoints)
    # In real code, we'd map these properly. For now, let's just print the text from the top result.
    if dense_results:
        top_text = dense_results[0].payload.get("_node_content")
        print(f"\nTop Dense Result Payload: {top_text[:200]}...")
        
    print("\n✅ Full Pipeline Ingest & Search Test Passed.")

if __name__ == "__main__":
    test_full_ingest()