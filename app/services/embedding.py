from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

embedding_model = OllamaEmbeddings(
    model=settings.embedding_model
)