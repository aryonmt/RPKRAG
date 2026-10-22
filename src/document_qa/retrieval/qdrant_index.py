from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from document_qa.core.config import settings
from document_qa.retrieval.embedding import LocalOnnxEmbedding

def get_qdrant_client() -> QdrantClient:
    """Returns a configured Qdrant client."""
    return QdrantClient(url=settings.QDRANT_URL)

def init_qdrant_collection() -> None:
    """Creates the collection and payload indexes if they don't exist."""
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE
            )
        )
        
        # Create payload indexes for fast filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="conversation_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Collection and indexes created.")
    else:
        print("Collection already exists.")

    for field_name in ("conversation_id", "document_id"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

def get_vector_store() -> QdrantVectorStore:
    """Returns a LlamaIndex QdrantVectorStore wrapper."""
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
    )

def delete_document_points(document_id: str) -> None:
    """Deletes all points belonging to a specific document."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )