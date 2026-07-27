import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_DB_PATH

def get_chroma_client():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

def get_or_create_collection(collection_name: str):
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def delete_collection(collection_name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        print(f"Collection {collection_name} deleted ✅")
    except Exception as e:
        print(f"Collection not found: {e}")

def collection_exists(collection_name: str) -> bool:
    client = get_chroma_client()
    collections = client.list_collections()
    return collection_name in collections