import uuid

import pytest

pytestmark = pytest.mark.integration
from document_qa.retrieval.qdrant_index import init_qdrant_collection, get_vector_store, get_qdrant_client
from document_qa.retrieval.embedding import FakeEmbedding
from llama_index.core.schema import TextNode
from qdrant_client.http import models
from document_qa.core.config import settings

def test_qdrant_insert_and_filter():
    print("--- Testing Qdrant ---")
    print("Initializing collection...")
    init_qdrant_collection()

    vector_store = get_vector_store()
    embed_model = FakeEmbedding()

    # Generate real UUIDs
    node_id_1 = str(uuid.uuid4())
    node_id_2 = str(uuid.uuid4())

    text_conv_1 = "درخواست مرخصی باید سه روز قبل داده شود."
    text_conv_2 = "حقوق مدیرعامل محرمانه است."
    
    emb_1 = embed_model._get_text_embedding(text_conv_1)
    emb_2 = embed_model._get_text_embedding(text_conv_2)

    node_1 = TextNode(
        id_=node_id_1,
        text=text_conv_1,
        embedding=emb_1,
        metadata={
            "conversation_id": "conv-A",
            "document_id": "doc-A",
        }
    )

    node_2 = TextNode(
        id_=node_id_2,
        text=text_conv_2,
        embedding=emb_2,
        metadata={
            "conversation_id": "conv-B",
            "document_id": "doc-B",
        }
    )

    print("Inserting nodes...")
    vector_store.add([node_1, node_2])
    print("Nodes inserted.")

    query = "مرخصی"
    query_emb = embed_model._get_query_embedding(query)

    client = get_qdrant_client()
    
    print(f"\nSearching for: '{query}' inside conversation 'conv-A'...")
    
    # Use query_points instead of the removed search method
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_emb,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(value="conv-A")
                )
            ]
        ),
        limit=10
    )

    # The results are inside response.points
    search_result = response.points

    print(f"Found {len(search_result)} results.")
    for point in search_result:
        print(f" - Point ID: {point.id}, Payload: {point.payload}")

    assert len(search_result) == 1, "Should only find 1 result in conv-A"
    assert str(search_result[0].id) == node_id_1, "Should be node_id_1"
    
    print("\n✅ Filtering works! Conversation B data was not leaked.")

if __name__ == "__main__":
    test_qdrant_insert_and_filter()