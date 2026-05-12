from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings

embedding_model = OllamaEmbeddings(
    model=settings.embedding_model
)


def get_vector_store(db_id: str):

    collection_name = f"db_{db_id}"
    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=settings.database_url,
        pre_delete_collection=False,
    )
    # IMPORTANT
    vector_store.create_collection()
    return vector_store

# def get_collection(db_id: str):
#     collection_name = f"db_{db_id}"
#     vector_store = PGVector(
#         embeddings=embedding_model,
#         collection_name=collection_name,
#         connection=settings.database_url,
#         pre_delete_collection=False,
#     )
#     return vector_store.get_collection()