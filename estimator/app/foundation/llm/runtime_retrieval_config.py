"""Runtime-mutable retrieval configuration, backed by Redis (Session 10).

Sibling of :mod:`runtime_config` (the LLM model knobs) but for the two retrieval
switches the live session toggles on the fly:

* ``search_mode`` — ``"vector"`` (dense k-NN only) or ``"hybrid"`` (dense +
  lexical FTS fused with RRF).
* ``rerank`` — whether the cross-encoder rescas the recall set (recall-then-rerank).

``Settings`` (.env) stays the immutable layer of *defaults* (``SEARCH_MODE`` /
``RERANKER_ENABLED``); this store holds runtime *overrides* in its own Redis hash
(``estimator:runtime_retrieval``, separate from the model-knobs hash) so the two
config surfaces never clobber each other.

Failure semantics mirror :class:`RuntimeModelConfig`:
- Reads degrade gracefully to the .env default if Redis is down.
- Writes re-raise (:class:`RuntimeConfigUnavailable`) so the API maps them to 503.
"""

from __future__ import annotations

import redis
import structlog

from app.config import Settings
from app.foundation.llm.runtime_config import RuntimeConfigUnavailable

log = structlog.get_logger()

HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "search_mode"
RERANK_KEY = "rerank"
RETRIEVAL_KEYS: tuple[str, ...] = (SEARCH_MODE_KEY, RERANK_KEY)

VALID_SEARCH_MODES: frozenset[str] = frozenset({"vector", "hybrid"})

# Re-export so callers can import the shared write-failure exception from here.
__all__ = [
    "HASH_KEY",
    "RETRIEVAL_KEYS",
    "RERANK_KEY",
    "SEARCH_MODE_KEY",
    "VALID_SEARCH_MODES",
    "RuntimeConfigUnavailable",
    "RuntimeRetrievalConfig",
]


def _bool_to_str(value: bool) -> str:
    return "true" if value else "false"


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the retrieval switches."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeRetrievalConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    # --- raw override access ------------------------------------------------

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    # --- search_mode -------------------------------------------------------

    def default_search_mode(self) -> str:
        return self._settings.SEARCH_MODE

    def effective_search_mode(self) -> str:
        """Override if set (and valid), else the .env default."""
        raw = self._get_raw(SEARCH_MODE_KEY)
        if raw in VALID_SEARCH_MODES:
            return raw
        return self.default_search_mode()

    def set_search_mode(self, value: str | None) -> None:
        """Set the search-mode override; ``None`` clears it (back to default)."""
        if value is not None and value not in VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search_mode: {value!r}")
        self._set(SEARCH_MODE_KEY, value)

    # --- rerank ------------------------------------------------------------

    def default_rerank(self) -> bool:
        return self._settings.RERANKER_ENABLED

    def effective_rerank(self) -> bool:
        raw = self._get_raw(RERANK_KEY)
        if raw is None:
            return self.default_rerank()
        return _str_to_bool(raw)

    def set_rerank(self, value: bool | None) -> None:
        """Set the rerank override; ``None`` clears it (back to default)."""
        self._set(RERANK_KEY, None if value is None else _bool_to_str(value))

    # --- shared write + snapshot ------------------------------------------

    def _set(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(HASH_KEY, key)
            else:
                self._redis.hset(HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def is_overridden(self, key: str) -> bool:
        return self._get_raw(key) is not None

    def snapshot(self) -> dict[str, dict[str, object]]:
        """``{search_mode, rerank}`` each as ``{effective, default, overridden}``."""
        try:
            overrides = self._redis.hgetall(HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            SEARCH_MODE_KEY: {
                "effective": self.effective_search_mode(),
                "default": self.default_search_mode(),
                "overridden": SEARCH_MODE_KEY in overrides,
            },
            RERANK_KEY: {
                "effective": self.effective_rerank(),
                "default": self.default_rerank(),
                "overridden": RERANK_KEY in overrides,
            },
        }

    def reset_all(self) -> None:
        try:
            self._redis.delete(HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc
