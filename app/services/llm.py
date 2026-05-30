from functools import lru_cache
import json
import logging
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)


def log_ollama_gpu_status(model: str | None = None) -> None:
    selected_model = model or settings.llm_model
    requested_gpu = settings.ollama_num_gpu is not None and settings.ollama_num_gpu > 0
    requested_gpu_count = settings.ollama_num_gpu or 0
    logger.info(
        "SQL agent model '%s' GPU request config: requested=%s, num_gpu=%s",
        selected_model,
        requested_gpu,
        requested_gpu_count,
    )

    ps_url = urljoin(settings.ollama_base_url.rstrip("/") + "/", "api/ps")
    try:
        request = Request(ps_url, method="GET")
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not query Ollama runtime status at %s: %s",
            ps_url,
            exc,
        )
        return

    models = payload.get("models", [])
    active = next(
        (
            item
            for item in models
            if str(item.get("name", "")).split(":")[0] == selected_model.split(":")[0]
        ),
        None,
    )
    if not active:
        logger.info(
            "Model '%s' not present in current Ollama process list; runtime GPU state unavailable until it is loaded.",
            selected_model,
        )
        return

    size_vram = int(active.get("size_vram") or 0)
    if size_vram > 0:
        logger.info(
            "Ollama runtime confirms GPU usage for '%s' (size_vram=%s bytes).",
            selected_model,
            size_vram,
        )
        return

    if requested_gpu:
        logger.warning(
            "GPU was requested for '%s' (num_gpu=%s), but Ollama reports no VRAM usage (size_vram=0). "
            "Server/runtime is likely not allowing GPU offload.",
            selected_model,
            requested_gpu_count,
        )
    else:
        logger.info(
            "GPU offload is not requested for '%s' and Ollama reports CPU-only runtime (size_vram=0).",
            selected_model,
        )


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
        base_url=settings.ollama_base_url,
        **options,
    )
