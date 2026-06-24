"""HTTP tests for GET/PUT /api/v1/config/retrieval (Session 10 switches)."""

from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_runtime_retrieval_config
from app.foundation.llm.runtime_retrieval_config import RuntimeRetrievalConfig
from app.main import app


def make_settings(**overrides) -> Settings:
    defaults = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test"}
    return Settings(_env_file=None, **{**defaults, **overrides})


@pytest.fixture
def fake_store():
    settings = make_settings()
    store = RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_store) -> TestClient:
    return TestClient(app)


def test_get_returns_snapshot_and_model(client) -> None:
    body = client.get("/api/v1/config/retrieval").json()

    assert body["retrieval"]["search_mode"] == {
        "effective": "vector",
        "default": "vector",
        "overridden": False,
    }
    assert body["retrieval"]["rerank"]["effective"] is False
    assert body["reranker_model"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert set(body["valid_search_modes"]) == {"vector", "hybrid"}


def test_put_overrides_search_mode_and_rerank(client, fake_store) -> None:
    response = client.put(
        "/api/v1/config/retrieval", json={"search_mode": "hybrid", "rerank": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval"]["search_mode"]["effective"] == "hybrid"
    assert body["retrieval"]["search_mode"]["overridden"] is True
    assert body["retrieval"]["rerank"]["effective"] is True
    assert fake_store.effective_search_mode() == "hybrid"
    assert fake_store.effective_rerank() is True


def test_put_partial_leaves_other_switch_untouched(client, fake_store) -> None:
    fake_store.set_rerank(True)

    client.put("/api/v1/config/retrieval", json={"search_mode": "hybrid"})

    assert fake_store.effective_search_mode() == "hybrid"
    assert fake_store.effective_rerank() is True  # untouched


def test_put_null_resets_to_default(client, fake_store) -> None:
    fake_store.set_search_mode("hybrid")

    response = client.put("/api/v1/config/retrieval", json={"search_mode": None})

    assert response.status_code == 200
    assert response.json()["retrieval"]["search_mode"]["overridden"] is False
    assert fake_store.effective_search_mode() == "vector"


def test_put_invalid_search_mode_is_422(client, fake_store) -> None:
    response = client.put("/api/v1/config/retrieval", json={"search_mode": "semantic"})

    assert response.status_code == 422
    assert "Invalid search_mode" in response.json()["detail"]
    assert fake_store.is_overridden("search_mode") is False


def test_put_empty_body_is_noop(client, fake_store) -> None:
    response = client.put("/api/v1/config/retrieval", json={})

    assert response.status_code == 200
    assert fake_store.is_overridden("search_mode") is False
    assert fake_store.is_overridden("rerank") is False
