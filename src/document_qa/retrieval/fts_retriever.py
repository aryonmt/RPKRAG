from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from document_qa.storage.sqlite import get_db_connection
from document_qa.retrieval.fts_query import build_fts_query
from document_qa.core.config import settings

class FtsRetriever(BaseRetriever):
    def __init__(self, conversation_id: str, ready_document_ids: list[str], **kwargs):
        self.conversation_id = conversation_id
        self.ready_document_ids = ready_document_ids
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not self.ready_document_ids:
            return []

        # Extract the string from the QueryBundle
        query_str = query_bundle.query_str
        
        fts_expr = build_fts_query(query_str)
        if not fts_expr:
            return []

        # Build the IN clause for document_ids safely
        placeholders = ",".join(["?"] * len(self.ready_document_ids))
        
        sql = f"""
            SELECT c.node_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.node_id = chunks_fts.node_id
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
              AND d.conversation_id = ?
              AND d.status IN ('indexed', 'partial_warning')
              AND c.document_id IN ({placeholders})
            ORDER BY score ASC
            LIMIT ?
        """
        
        params = [fts_expr, self.conversation_id] + self.ready_document_ids + [settings.RETRIEVAL_LEXICAL_TOP_K]

        conn = get_db_connection()
        try:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            node = TextNode(id_=row["node_id"], text="") # Text will be hydrated later
            # bm25 returns negative scores where smaller is better.
            # We negate it so that a higher score means a better match (consistent with vector search)
            score = -row["score"]
            results.append(NodeWithScore(node=node, score=score))

        return results