"""HTTP layer for the runtime model configuration (Settings UI).

Thin router: validation of the partial update is the only logic here — the
override store lives in ``app/foundation/llm/runtime_config.py``. Changing a
model takes effect on the NEXT LLM call (wrapper/service read the store per
call); nothing is rebuilt and no restart is needed.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.dependencies import get_runtime_config, get_runtime_retrieval_config
from app.foundation.llm.runtime_config import (
    MODEL_KEYS,
    RuntimeConfigUnavailable,
    RuntimeModelConfig,
)
from app.foundation.llm.runtime_retrieval_config import (
    VALID_SEARCH_MODES,
    RuntimeRetrievalConfig,
)
from app.foundation.llm.wrapper import _provider_from_model

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/config", tags=["config"])

EMBEDDING_MODEL_NOTE = "Read-only: changing it would invalidate all stored vectors."

PROVIDER_KEY_FIELDS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ModelUpdateRequest(BaseModel):
    """Partial update: only the keys present are touched; ``null`` resets."""

    models: dict[str, str | None] = Field(min_length=1)


def _available_models(settings: Settings) -> list[str]:
    """The catalog, filtered by the API keys actually configured."""
    available = []
    for model in settings.AVAILABLE_MODELS:
        key_field = PROVIDER_KEY_FIELDS.get(_provider_from_model(model))
        if key_field and getattr(settings, key_field):
            available.append(model)
    return available


def _config_payload(runtime_config: RuntimeModelConfig, settings: Settings) -> dict:
    return {
        "models": runtime_config.snapshot(),
        "available_models": _available_models(settings),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_model_note": EMBEDDING_MODEL_NOTE,
    }


@router.get("/models")
def get_models(
    runtime_config: RuntimeModelConfig = Depends(get_runtime_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Current model configuration: effective/default/overridden per knob."""
    return _config_payload(runtime_config, settings)


@router.put("/models")
def update_models(
    request: ModelUpdateRequest,
    runtime_config: RuntimeModelConfig = Depends(get_runtime_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Apply a partial override update. All-or-nothing: every key/value in the
    payload is validated BEFORE anything is written."""
    available = _available_models(settings)

    # Validate everything first — a bad entry must not half-apply the batch.
    for key, value in request.models.items():
        if key not in MODEL_KEYS:
            raise HTTPException(status_code=422, detail=f"Unknown model key: {key}")
        if value is None:
            continue  # reset is always valid
        if value not in settings.AVAILABLE_MODELS:
            raise HTTPException(status_code=422, detail=f"Model '{value}' is not in the catalog")
        if value not in available:
            key_field = PROVIDER_KEY_FIELDS.get(_provider_from_model(value), "API key")
            raise HTTPException(
                status_code=400,
                detail=f"Model '{value}' requires {key_field}, which is not configured",
            )

    try:
        for key, value in request.models.items():
            old_effective = runtime_config.effective(key)
            runtime_config.set(key, value)
            log.info(
                "runtime_config_changed",
                key=key,
                old_effective=old_effective,
                new_value=value,
                reset=value is None,
            )
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_config_write_failed", error=str(exc)[:200])
        raise HTTPException(status_code=503, detail="Runtime config store unavailable") from exc

    return _config_payload(runtime_config, settings)


# --- Session 10: hybrid retrieval + reranking switches ----------------------
# A second, separate config surface (its own Redis hash) for the retrieval
# toggles. Like the model knobs, changes take effect on the NEXT retrieval call
# — nothing is rebuilt, no restart needed.

class RetrievalConfigUpdate(BaseModel):
    """Partial update of the retrieval switches.

    Omitting a field leaves it untouched; sending it as ``null`` resets that
    switch to its .env default; sending a value sets the override. Omitted vs
    explicit-null is told apart with ``model_fields_set`` (both arrive as
    ``None`` otherwise)."""

    search_mode: str | None = None
    rerank: bool | None = None


def _retrieval_payload(runtime: RuntimeRetrievalConfig, settings: Settings) -> dict:
    return {
        "retrieval": runtime.snapshot(),
        "reranker_model": settings.RERANKER_MODEL,
        "valid_search_modes": sorted(VALID_SEARCH_MODES),
    }


@router.get("/retrieval")
def get_retrieval_config(
    runtime: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Current retrieval switches: effective/default/overridden + reranker model."""
    return _retrieval_payload(runtime, settings)


@router.put("/retrieval")
def update_retrieval_config(
    request: RetrievalConfigUpdate,
    runtime: RuntimeRetrievalConfig = Depends(get_runtime_retrieval_config),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Apply a partial override of the retrieval switches.

    Validates ``search_mode`` against the allowed set (422 on a bad value)
    before writing; a Redis write failure maps to 503."""
    fields_set = request.model_fields_set
    if request.search_mode is not None and request.search_mode not in VALID_SEARCH_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid search_mode '{request.search_mode}'. "
            f"Expected one of {sorted(VALID_SEARCH_MODES)}.",
        )

    try:
        if "search_mode" in fields_set:
            runtime.set_search_mode(request.search_mode)
            log.info("runtime_retrieval_changed", key="search_mode", new_value=request.search_mode)
        if "rerank" in fields_set:
            runtime.set_rerank(request.rerank)
            log.info("runtime_retrieval_changed", key="rerank", new_value=request.rerank)
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_retrieval_write_failed", error=str(exc)[:200])
        raise HTTPException(status_code=503, detail="Runtime config store unavailable") from exc

    return _retrieval_payload(runtime, settings)
