"""Unit tests for the Redis-backed runtime retrieval configuration (Session 10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest
import redis as redis_lib

from app.config import Settings
from app.foundation.llm.runtime_retrieval_config import (
    HASH_KEY,
    RuntimeConfigUnavailable,
    RuntimeRetrievalConfig,
)


def make_settings(**overrides) -> Settings:
    return Settings(OPENAI_API_KEY="sk-test", _env_file=None, **overrides)


@pytest.fixture
def store() -> RuntimeRetrievalConfig:
    return RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), make_settings())


def test_defaults_come_from_settings(store) -> None:
    assert store.effective_search_mode() == "vector"
    assert store.effective_rerank() is False
    assert store.is_overridden("search_mode") is False
    assert store.is_overridden("rerank") is False


def test_settings_default_flows_through(store) -> None:
    custom = RuntimeRetrievalConfig(
        fakeredis.FakeRedis(decode_responses=True),
        make_settings(SEARCH_MODE="hybrid", RERANKER_ENABLED=True),
    )
    assert custom.effective_search_mode() == "hybrid"
    assert custom.effective_rerank() is True


def test_set_and_effective_round_trip(store) -> None:
    store.set_search_mode("hybrid")
    store.set_rerank(True)
    assert store.effective_search_mode() == "hybrid"
    assert store.effective_rerank() is True
    assert store.is_overridden("search_mode") is True
    assert store.is_overridden("rerank") is True


def test_set_none_resets_to_default(store) -> None:
    store.set_search_mode("hybrid")
    store.set_search_mode(None)
    assert store.effective_search_mode() == "vector"
    assert store.is_overridden("search_mode") is False


def test_invalid_search_mode_raises(store) -> None:
    with pytest.raises(ValueError, match="Invalid search_mode"):
        store.set_search_mode("semantic")


def test_rerank_truthy_strings_parse(store) -> None:
    store._redis.hset(HASH_KEY, "rerank", "true")
    assert store.effective_rerank() is True
    store._redis.hset(HASH_KEY, "rerank", "false")
    assert store.effective_rerank() is False


def test_snapshot_shape(store) -> None:
    store.set_rerank(True)
    snapshot = store.snapshot()
    assert set(snapshot) == {"search_mode", "rerank"}
    assert snapshot["rerank"] == {"effective": True, "default": False, "overridden": True}
    assert snapshot["search_mode"]["overridden"] is False


def test_reset_all_clears_overrides(store) -> None:
    store.set_search_mode("hybrid")
    store.set_rerank(True)
    store.reset_all()
    assert store.effective_search_mode() == "vector"
    assert store.effective_rerank() is False


def test_reads_degrade_to_defaults_when_redis_down() -> None:
    broken = MagicMock()
    broken.hget.side_effect = redis_lib.RedisError("down")
    broken.hgetall.side_effect = redis_lib.RedisError("down")
    store = RuntimeRetrievalConfig(broken, make_settings())

    assert store.effective_search_mode() == "vector"
    assert store.effective_rerank() is False
    assert store.snapshot()["search_mode"]["overridden"] is False


def test_writes_raise_when_redis_down() -> None:
    broken = MagicMock()
    broken.hset.side_effect = redis_lib.RedisError("down")
    store = RuntimeRetrievalConfig(broken, make_settings())

    with pytest.raises(RuntimeConfigUnavailable):
        store.set_search_mode("hybrid")
