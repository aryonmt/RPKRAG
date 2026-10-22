import uuid

import pytest

pytestmark = [pytest.mark.live, pytest.mark.model]
from pathlib import Path
from document_qa.storage.sqlite import get_db_connection
from document_qa.generation.answerer import LLMService
from document_qa.retrieval.embedding import LocalOnnxEmbedding
from document_qa.retrieval.qdrant_index import get_qdrant_client
from qdrant_client.http import models
from document_qa.core.config import settings

def test_llm_answer():
    print("--- Testing LLM Answer Generation ---")
    
    conn = get_db_connection()
    # Find a document that actually has chunks (i.e., the real PDF we ingested)
    doc_row = conn.execute("""
        SELECT id, conversation_id, filename 
        FROM documents 
        WHERE chunk_count > 0 
        ORDER BY created_at DESC LIMIT 1
    """).fetchone()

    if not doc_row:
        conn.close()
        pytest.skip("No ingested document with chunks")

    conv_id = doc_row["conversation_id"]
    doc_id = doc_row["id"]
    print(f"Using conversation: {conv_id} (Document: {doc_row['filename']})")

    # 1. Get evidence from Qdrant (Simulating the Hybrid Search)
    embed_model = LocalOnnxEmbedding()
    client = get_qdrant_client()
    query = "تمرین شماره ۱ در مورد چیست؟"
    
    q_emb = embed_model._get_query_embedding(query)
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=q_emb,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="conversation_id", match=models.MatchValue(value=conv_id))]
        ),
        limit=3
    )
    
    # Hydrate evidence from SQLite
    evidence_chunks = []
    for point in response.points:
        node_id = point.id
        row = conn.execute("SELECT * FROM chunks WHERE node_id = ?", (str(node_id),)).fetchone()
        if row:
            evidence_chunks.append(dict(row))

    conn.close()

    print(f"Found {len(evidence_chunks)} evidence chunks.")
    
    if not evidence_chunks:
        pytest.fail("No evidence found in Qdrant")

    # 2. Call LLM
    try:
        llm_service = LLMService()
        print("Sending to LLM (this may take a few seconds)...")
        result = llm_service.generate_answer(query, evidence_chunks)
        
        print("\n--- LLM Result ---")
        print(f"Status: {result['status']}")
        
        if result.get('claims'):
            print("Claims (Validated):")
            for c in result['claims']:
                print(f" - {c}")
        else:
            print("No valid claims survived validation.")
            
        if result.get('citations'):
            print(f"Citations: {result['citations']}")
            
        print("\n✅ LLM Generation & Validation Test Completed.")
        
    except ValueError as e:
        if "llm_not_configured" in str(e):
            pytest.skip("LLM is not configured")
        raise
    except Exception as e:
        pytest.fail(f"LLM call failed: {e}")

if __name__ == "__main__":
    test_llm_answer()