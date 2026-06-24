"""Unit tests for the hybrid/rerank retrieval pipeline (Session 10).

The vector store and the cross-encoder are faked (no Postgres, no torch). We
assert the four (search_mode x rerank) configurations end-to-end: recall width
selection, RRF fusion order for hybrid, recall-then-rerank truncation, the
soft-fail contract, and that the lexical branch is queried only for hybrid."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.dependencies as deps
from app.generation.rag.retrieval import retrieve, retrieve_ranked
from app.generation.rag.retrieval.pipeline import Candidate


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _factory():
    return FakeSession()


def _vrow(chunk_id: int, distance: float, budget_id: str = "BUD-X") -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"content-{chunk_id}",
        metadata_={"client_sector": "finance", "year": 2024, "budget_id": budget_id},
        distance=distance,
    )


def _lrow(chunk_id: int, rank: float, budget_id: str = "BUD-X") -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"content-{chunk_id}",
        metadata_={"client_sector": "finance", "year": 2024, "budget_id": budget_id},
        rank=rank,
    )


class FakeStore:
    def __init__(self, *, vector_rows, lexical_rows, candidates=20):
        self._vector_rows = vector_rows
        self._lexical_rows = lexical_rows
        self._candidates = candidates
        self.vector_calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    async def search_filtered(self, session, **kwargs):
        self.vector_calls.append(kwargs)
        return self._vector_rows, self._candidates

    async def search_lexical(self, session, **kwargs):
        self.lexical_calls.append(kwargs)
        return self._lexical_rows


class FakeReranker:
    """Orders candidates by a fixed id→score map (higher first)."""

    def __init__(self, scores: dict[int, float]):
        self._scores = scores
        self.calls: list[dict] = []

    def rerank(self, query, candidates, *, top_n):
        self.calls.append({"query": query, "ids": [c.id for c in candidates], "top_n": top_n})
        ordered = sorted(candidates, key=lambda c: self._scores.get(c.id, 0.0), reverse=True)
        return ordered[:top_n]


@pytest.fixture
def wire(monkeypatch):
    def _wire(store):
        monkeypatch.setattr(deps, "get_async_session_factory", lambda: _factory)
        monkeypatch.setattr(deps, "get_chunk_store", lambda: store)

    return _wire


async def test_config_a_vector_no_rerank_uses_exact_top_k(wire) -> None:
    store = FakeStore(vector_rows=[_vrow(1, 0.1), _vrow(2, 0.2), _vrow(3, 0.3)], lexical_rows=[])
    wire(store)

    candidates, evaluated = await retrieve_ranked(
        query_embedding=[0.0] * 1536,
        query_text="banking payments",
        search_mode="vector",
        rerank=False,
        top_k=2,
    )

    assert [c.id for c in candidates] == [1, 2]  # exact top_k, distance order
    assert evaluated == 20
    # Not wide: vector limit == top_k, and the lexical branch was never touched.
    assert store.vector_calls[0]["top_k"] == 2
    assert store.lexical_calls == []


async def test_config_b_hybrid_fuses_with_rrf(wire) -> None:
    # Vector order: 1,2,3 ; lexical order: 3,2,4. RRF should boost agreement.
    store = FakeStore(
        vector_rows=[_vrow(1, 0.1), _vrow(2, 0.2), _vrow(3, 0.3)],
        lexical_rows=[_lrow(3, 0.9), _lrow(2, 0.5), _lrow(4, 0.2)],
    )
    wire(store)

    candidates, _ = await retrieve_ranked(
        query_embedding=[0.0] * 1536,
        query_text="banking payments",
        search_mode="hybrid",
        rerank=False,
        top_k=10,
        recall_k=50,
        rrf_k=60,
    )

    ids = [c.id for c in candidates]
    # 2 (ranks 2 & 2) and 3 (ranks 3 & 1) appear in both → above 1 and 4.
    assert set(ids[:2]) == {2, 3}
    assert ids[-1] == 4 or ids.index(4) > ids.index(1)
    # Hybrid is "wide": both branches queried at recall_k.
    assert store.vector_calls[0]["top_k"] == 50
    assert store.lexical_calls[0]["top_k"] == 50
    # Lexical-only hit (4) carries no cosine distance.
    cand4 = next(c for c in candidates if c.id == 4)
    assert cand4.distance is None


async def test_config_c_vector_rerank_recall_then_rerank(wire) -> None:
    store = FakeStore(
        vector_rows=[_vrow(i, 0.01 * i) for i in range(1, 6)],
        lexical_rows=[],
    )
    wire(store)
    reranker = FakeReranker(scores={5: 1.0, 1: 0.9, 3: 0.5, 2: 0.1, 4: 0.0})

    candidates, _ = await retrieve_ranked(
        query_embedding=[0.0] * 1536,
        query_text="banking payments",
        search_mode="vector",
        rerank=True,
        top_k=5,
        recall_k=50,
        rerank_top_n=2,
        reranker=reranker,
    )

    # Reranker reorders and truncates to top_n=2.
    assert [c.id for c in candidates] == [5, 1]
    # Broad recall was fetched (recall_k), not top_k.
    assert store.vector_calls[0]["top_k"] == 50
    assert reranker.calls[0]["top_n"] == 2
    assert store.lexical_calls == []  # vector mode → no lexical branch


async def test_config_d_hybrid_rerank_queries_both_then_reranks(wire) -> None:
    store = FakeStore(
        vector_rows=[_vrow(1, 0.1), _vrow(2, 0.2)],
        lexical_rows=[_lrow(3, 0.9), _lrow(2, 0.5)],
    )
    wire(store)
    reranker = FakeReranker(scores={3: 1.0, 1: 0.5, 2: 0.2})

    candidates, _ = await retrieve_ranked(
        query_embedding=[0.0] * 1536,
        query_text="banking payments",
        search_mode="hybrid",
        rerank=True,
        top_k=5,
        recall_k=50,
        rerank_top_n=3,
        reranker=reranker,
    )

    assert [c.id for c in candidates] == [3, 1, 2]
    assert store.vector_calls[0]["top_k"] == 50
    assert store.lexical_calls[0]["top_k"] == 50


async def test_retrieve_projects_to_retrieval_result_and_soft_fail(wire) -> None:
    empty = FakeStore(vector_rows=[], lexical_rows=[], candidates=7)
    wire(empty)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="nothing matches",
        search_mode="vector",
        rerank=False,
    )

    assert result.chunks == []
    assert result.low_confidence is True
    assert result.candidates_evaluated == 7


async def test_retrieve_result_carries_flattened_metadata(wire) -> None:
    store = FakeStore(
        vector_rows=[_vrow(1, 0.1, budget_id="BUD-2024-003")],
        lexical_rows=[],
    )
    wire(store)

    result = await retrieve(
        query_embedding=[0.0] * 1536,
        query_text="payments gateway",
        search_mode="vector",
        rerank=False,
        top_k=5,
    )

    chunk = result.chunks[0]
    assert chunk.sector == "finance"
    assert chunk.project_year == 2024
    assert chunk.distance == pytest.approx(0.1)


def test_candidate_budget_id_from_metadata() -> None:
    c = Candidate(
        id=1, document_id=10, chunk_type="budget_component", content="x",
        metadata={"budget_id": "BUD-2024-001", "client_sector": "finance", "year": 2024},
    )
    assert c.budget_id == "BUD-2024-001"
    assert c.sector == "finance"
    assert c.project_year == 2024
