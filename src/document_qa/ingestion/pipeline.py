from pathlib import Path
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from document_qa.ingestion.extractor import extract_document
from document_qa.ingestion.cleaner import clean_pages
from document_qa.ingestion.chunker import LocalTokenizer, chunk_pages
from document_qa.retrieval.embedding import LocalOnnxEmbedding
from document_qa.retrieval.qdrant_index import get_vector_store
from document_qa.storage.sqlite import get_db_connection
from document_qa.storage.repositories import save_document_chunks
from document_qa.core.config import settings
from document_qa.ingestion.normalizer import normalize_for_search

def process_document(
    file_path: Path,
    document_id: str,
    conversation_id: str,
    filename: str,
    title: str,
    generation: int = 1
):
    print(f"Starting ingest for {filename}...")

    ext = file_path.suffix.lower()
    pages = extract_document(file_path, ext)
    cleaned_pages, clean_warnings = clean_pages(pages)
    
    tokenizer = LocalTokenizer(settings.EMBEDDING_MODEL_DIR)
    chunks = chunk_pages(
        pages=cleaned_pages,
        document_id=document_id,
        filename=filename,
        title=title,
        tokenizer=tokenizer
    )
    
    if not chunks:
        raise ValueError("no_extractable_text")
        
    print(f"Created {len(chunks)} chunks.")

    for chunk in chunks:
        chunk["conversation_id"] = conversation_id
        chunk["document_id"] = document_id

    print("Generating embeddings and uploading to Qdrant...")
    embed_model = LocalOnnxEmbedding()
    vector_store = get_vector_store()
    
    nodes = []
    for chunk in chunks:
        embed_input = normalize_for_search((chunk["breadcrumb"] or "") + "\n" + chunk["original_text"])
        emb = embed_model._get_text_embedding(embed_input)
        
        node = TextNode(
            id_=chunk["node_id"],
            text=chunk["original_text"],
            embedding=emb,
            metadata={
                "conversation_id": conversation_id,
                "document_id": document_id,
                "ordinal": chunk["ordinal"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "filename": filename,
                "title": title,
                "breadcrumb": chunk["breadcrumb"]
            }
        )
        
        # FIXED: Proper way to set document ID in LlamaIndex
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document_id)
        
        node.excluded_embed_metadata_keys = list(node.metadata.keys())
        node.excluded_llm_metadata_keys = list(node.metadata.keys())
        nodes.append(node)

    batch_size = 64
    for i in range(0, len(nodes), batch_size):
        vector_store.add(nodes[i:i+batch_size])
    print("Qdrant upload complete.")

    conn = get_db_connection()
    try:
        success = save_document_chunks(
            conn=conn,
            document_id=document_id,
            chunks=chunks,
            page_count=len(pages),
            warnings=clean_warnings,
            expected_generation=generation
        )
        if not success:
            from document_qa.retrieval.qdrant_index import delete_document_points
            try:
                delete_document_points(document_id)
            except Exception as cleanup_error:
                print(f"Failed to drop stale Qdrant points for {document_id}: {cleanup_error}")
            raise RuntimeError("generation_mismatch")
        print("SQLite save complete.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    print("Ingest finished successfully.")