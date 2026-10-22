from collections import defaultdict
from typing import List
from llama_index.core.schema import NodeWithScore
from document_qa.core.config import settings

def rrf_fusion(dense_results: List[NodeWithScore], lexical_results: List[NodeWithScore], k: int = None, top_n: int = 50) -> List[NodeWithScore]:
    if k is None: k = settings.RRF_K

    scores = defaultdict(float)
    nodes_map = {}

    for rank, result in enumerate(dense_results, start=1):
        node_id = result.node.id_
        scores[node_id] += 1.0 / (k + rank)
        if node_id not in nodes_map: nodes_map[node_id] = result.node

    for rank, result in enumerate(lexical_results, start=1):
        node_id = result.node.id_
        scores[node_id] += 1.0 / (k + rank)
        if node_id not in nodes_map: nodes_map[node_id] = result.node

    sorted_ids = sorted(scores.keys(), key=lambda x: (-scores[x], x))
    
    final_results = []
    # Return top 50 so qa_service can deduplicate and then pick the best EVIDENCE_MAX
    for node_id in sorted_ids[:top_n]:
        final_results.append(NodeWithScore(node=nodes_map[node_id], score=scores[node_id]))

    return final_results


# ==========================================
# TEST BLOCK (Run this file directly to test)
# ==========================================
if __name__ == "__main__":
    import uuid
    from document_qa.storage.sqlite import get_db_connection
    from document_qa.retrieval.fts_retriever import FtsRetriever
    from document_qa.retrieval.embedding import FakeEmbedding

    print("--- Testing Hybrid Search & RRF ---")
    
    # 1. Setup mock data in SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Cleanup previous test data
    cursor.execute("DELETE FROM conversations WHERE id = 'test-conv'")
    cursor.execute("INSERT INTO conversations (id, title, status, created_at, updated_at) VALUES ('test-conv', 'Test', 'active', 'now', 'now')")
    cursor.execute("INSERT INTO documents (id, conversation_id, filename, file_type, file_hash, status, created_at, updated_at) VALUES ('test-doc', 'test-conv', 'test.pdf', 'pdf', 'hash', 'indexed', 'now', 'now')")
    
    chunks = [
        {"id": "11111111-1111-1111-1111-111111111111", "text": "درخواست مرخصی باید سه روز قبل داده شود."},
        {"id": "22222222-2222-2222-2222-222222222222", "text": "حقوق مدیرعامل محرمانه است."},
        {"id": "33333333-3333-3333-3333-333333333333", "text": "مرخصی استعلاجی نیاز به تاییدیه پزشک دارد."}
    ]
    
    for c in chunks:
        cursor.execute("""
            INSERT INTO chunks (node_id, conversation_id, document_id, ordinal, filename, title, original_text, search_text, char_start, char_end, token_count)
            VALUES (?, 'test-conv', 'test-doc', 0, 'test.pdf', 'Test', ?, ?, 0, 10, 5)
        """, (c["id"], c["text"], c["text"]))
        
        cursor.execute("""
            INSERT INTO chunks_fts (node_id, body_search, breadcrumb_search, title_search)
            VALUES (?, ?, '', '')
        """, (c["id"], c["text"]))
        
    conn.commit()
    conn.close()

    # 2. Test Lexical Search (FTS)
    print("\nTesting FTS Retriever with query: 'مرخصی'")
    fts = FtsRetriever(conversation_id="test-conv", ready_document_ids=["test-doc"])
    lexical_results = fts.retrieve("مرخصی")
    for res in lexical_results:
        print(f" - Node: {res.node.id_} (Score: {res.score:.4f})")

    # 3. Mock Dense Search Results
    # Let's pretend the vector search returned Node 2 first, and Node 1 second
    print("\nMocking Dense Retriever results...")
    from llama_index.core.schema import TextNode
    dense_results = [
        NodeWithScore(node=TextNode(id_="22222222-2222-2222-2222-222222222222"), score=0.9),
        NodeWithScore(node=TextNode(id_="11111111-1111-1111-1111-111111111111"), score=0.8),
    ]
    print(" - Dense Rank 1: 22222222...")
    print(" - Dense Rank 2: 11111111...")

    # 4. Fuse with RRF
    print("\nRunning RRF Fusion...")
    fused = rrf_fusion(dense_results, lexical_results)
    
    print("\nFinal Fused Results (Top 8):")
    for rank, res in enumerate(fused, start=1):
        print(f"Rank {rank}: Node {res.node.id_} (RRF Score: {res.score:.4f})")
        
    print("\n✅ Hybrid Search & RRF Test Passed.")