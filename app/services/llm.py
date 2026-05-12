from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core.config import settings


@lru_cache(maxsize=4)
def get_llm(model: str | None = None) -> ChatOllama:
    options = {
        "temperature": settings.llm_temperature,
        "keep_alive": settings.ollama_keep_alive,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": settings.ollama_num_predict,
        "sync_client_kwargs": {"timeout": settings.ollama_request_timeout_seconds},
    }
    if settings.ollama_num_gpu is not None:
        options["num_gpu"] = settings.ollama_num_gpu
    if settings.ollama_num_thread is not None:
        options["num_thread"] = settings.ollama_num_thread

    return ChatOllama(
        model=model or settings.llm_model,
        **options,
    )
