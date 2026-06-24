"""Unit tests for the cross-encoder reranker (Session 10).

The real ``sentence_transformers.CrossEncoder`` (torch weights) is never loaded:
we inject a fake CrossEncoder via the module path the lazy loader imports. We
assert lazy loading (no load at construction, loaded on first rerank, loaded
once), score-descending ordering, top-n truncation and the empty short-circuit
that must NOT trigger a model load."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from app.generation.rag.retrieval.reranker import CrossEncoderReranker


@dataclass
class _Doc:
    id: int
    content: str


class _FakeCrossEncoder:
    """Scores a pair by the (test-controlled) score baked into the content tag.

    Content format: ``"score=<float>::<text>"``. Records load + predict calls."""

    instances: list["_FakeCrossEncoder"] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.predict_calls: list[list[tuple[str, str]]] = []
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs):
        self.predict_calls.append(list(pairs))
        return [float(doc.split("score=")[1].split("::")[0]) for _q, doc in pairs]


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Inject a fake ``sentence_transformers`` module with our CrossEncoder."""
    _FakeCrossEncoder.instances.clear()
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _FakeCrossEncoder


def _doc(doc_id: int, score: float, text: str = "doc") -> _Doc:
    return _Doc(id=doc_id, content=f"score={score}::{text}")


def test_not_loaded_at_construction(fake_sentence_transformers) -> None:
    reranker = CrossEncoderReranker("fake-model")
    assert reranker.is_loaded is False
    assert fake_sentence_transformers.instances == []


def test_loads_on_first_rerank_and_orders_by_score(fake_sentence_transformers) -> None:
    reranker = CrossEncoderReranker("fake-model")
    docs = [_doc(1, 0.1), _doc(2, 0.9), _doc(3, 0.5)]

    ranked = reranker.rerank("q", docs, top_n=3)

    assert reranker.is_loaded is True
    assert [d.id for d in ranked] == [2, 3, 1]  # descending score


def test_top_n_truncates(fake_sentence_transformers) -> None:
    reranker = CrossEncoderReranker("fake-model")
    docs = [_doc(1, 0.1), _doc(2, 0.9), _doc(3, 0.5), _doc(4, 0.7)]

    ranked = reranker.rerank("q", docs, top_n=2)

    assert [d.id for d in ranked] == [2, 4]


def test_model_loaded_only_once_across_calls(fake_sentence_transformers) -> None:
    reranker = CrossEncoderReranker("fake-model")
    reranker.rerank("q", [_doc(1, 0.5)], top_n=1)
    reranker.rerank("q2", [_doc(2, 0.5)], top_n=1)

    assert len(fake_sentence_transformers.instances) == 1


def test_empty_candidates_short_circuit_without_loading(fake_sentence_transformers) -> None:
    reranker = CrossEncoderReranker("fake-model")

    assert reranker.rerank("q", [], top_n=5) == []
    assert reranker.is_loaded is False
    assert fake_sentence_transformers.instances == []
