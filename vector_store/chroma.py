import logging
import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_DB_PATH
from typing import Optional

logger = logging.getLogger("chroma")


def get_chroma_client():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

def get_or_create_collection(collection_name: str, metadata: Optional[dict] = None):
    """
    Get or create a ChromaDB collection.
    Accepts optional metadata dict that is stored on the collection
    (e.g. schema_version for auto-reindex detection).
    """
    client = get_chroma_client()

    # Build collection metadata — always include cosine distance
    collection_meta = {"hnsw:space": "cosine"}
    if metadata:
        collection_meta.update(metadata)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=collection_meta
    )
    return collection

def delete_collection(collection_name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        logger.info("Collection %s deleted ✅", collection_name)
    except Exception as e:
        logger.warning("Collection not found: %s", e)

def collection_exists(collection_name: str) -> bool:
    client = get_chroma_client()
    collections = client.list_collections()
    return collection_name in collections

def get_collection_metadata(collection_name: str) -> Optional[dict]:
    """
    Retrieve the metadata dict stored on a collection.
    Used to check schema_version for auto-reindex decisions.
    Returns None if the collection doesn't exist.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
        return collection.metadata
    except Exception:
        return None