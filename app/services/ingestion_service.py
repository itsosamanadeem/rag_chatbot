import hashlib

from app.services.schema_extractor import extract_schema_documents
from app.services.vector_store import get_vector_store


def generate_db_hash(db_url: str):
    return hashlib.md5(db_url.encode()).hexdigest()


def ingest_database(db_url: str):

    db_id = generate_db_hash(db_url)

    docs = extract_schema_documents(db_url)

    vector_store = get_vector_store(db_id)

    vector_store.add_documents(docs)

    return {
        "db_id": db_id,
        "documents_indexed": len(docs),
    }