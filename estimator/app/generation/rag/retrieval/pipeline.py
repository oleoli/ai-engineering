"""Single retrieval entry point: hybrid recall + optional reranking (Session 10).

``retrieve()`` composes the **four** retrieval configurations behind two
switches the caller resolves by precedence (request param → runtime override →
.env default) and passes in already-resolved:

| Config | ``search_mode`` | ``rerank`` | Behaviour                                  |
|--------|-----------------|------------|--------------------------------------------|
| **A**  | ``vector``      | ``False``  | dense k-NN, top-k (the Session 9 baseline) |
| **B**  | ``hybrid``      | ``False``  | dense + lexical fused with RRF, top-k      |
| **C**  | ``vector``      | ``True``   | broad dense recall → cross-encoder → top-n |
| **D**  | ``hybrid``      | ``True``   | broad hybrid recall → cross-encoder → top-n|

Passing the switches explicitly (instead of reading the runtime store here) is
what makes the four configs *reproducibly invocable* — the eval harness pins
each config regardless of whatever the live toggles happen to be.

The dense branch keeps the Session 9 relevance threshold (so config A's
soft-fail is byte-for-byte the old behaviour); the lexical branch contributes
ordering only. Reranking, when on, is the sole truncation to ``top_n``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import structlog

from app.config import get_settings
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk

log = structlog.get_logger()

SearchMode = Literal["vector", "hybrid"]


@dataclass
class Candidate:
    """A retrieved chunk inside the pipeline, before it is projected to the API.

    Carries the full ``metadata`` dict (so ``budget_id`` survives for the eval
    harness and for citations) plus the per-branch signals (``distance`` for the
    dense branch, ``rerank_score`` once the cross-encoder has scored it)."""

    id: int
    document_id: int
    chunk_type: str
    content: str
    metadata: dict = field(default_factory=dict)
    distance: float | None = None
    rerank_score: float | None = None

    @property
    def budget_id(self) -> str | None:
        return self.metadata.get("budget_id")

    @property
    def sector(self) -> str:
        return str(self.metadata.get("client_sector", "unknown"))

    @property
    def project_year(self) -> int:
        return int(self.metadata.get("year", 0) or 0)

    def to_retrieved_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            id=self.id,
            content=self.content,
            sector=self.sector,
            project_year=self.project_year,
            chunk_type=self.chunk_type,
            # Cosine distance when the chunk came (also) from the dense branch;
            # a lexical-only hit has no distance, surfaced as the 1.0 far value.
            distance=self.distance if self.distance is not None else 1.0,
        )


def _candidate_from_row(row, *, distance: float | None) -> Candidate:
    return Candidate(
        id=row.id,
        document_id=row.document_id,
        chunk_type=row.chunk_type,
        content=row.content,
        metadata=row.metadata_,
        distance=distance,
    )


async def retrieve_ranked(
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: SearchMode,
    rerank: bool,
    top_k: int | None = None,
    recall_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    distance_threshold: float | None = None,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker=None,
) -> tuple[list[Candidate], int]:
    """Run the resolved retrieval configuration; return ``(candidates, evaluated)``.

    ``candidates`` are best-first. ``evaluated`` is how many chunks matched the
    dense branch's structural filters (the Session 9 ``candidates_evaluated``).
    """
    import asyncio

    from app.dependencies import (
        get_async_session_factory,
        get_chunk_store,
        get_reranker,
    )

    settings = get_settings()
    top_k = settings.RETRIEVAL_TOP_K if top_k is None else top_k
    recall_k = settings.RETRIEVAL_RECALL_TOP_K if recall_k is None else recall_k
    rerank_top_n = settings.RERANK_TOP_N if rerank_top_n is None else rerank_top_n
    rrf_k = settings.RRF_K if rrf_k is None else rrf_k
    if distance_threshold is None:
        distance_threshold = settings.RETRIEVAL_DISTANCE_THRESHOLD

    is_hybrid = search_mode == "hybrid"
    # A wide recall is only worth paying for when something downstream will use
    # the extra candidates: either RRF fusion (hybrid) or the cross-encoder.
    wide = rerank or is_hybrid
    vector_limit = recall_k if wide else top_k

    session_factory = get_async_session_factory()
    store = get_chunk_store()
    filters = dict(
        sectors=sectors,
        project_year_min=project_year_min,
        project_year_max=project_year_max,
        chunk_types=chunk_types,
    )

    started = time.perf_counter()
    try:
        async with session_factory() as session:
            vector_rows, candidates_evaluated = await store.search_filtered(
                session,
                query_vector=query_embedding,
                top_k=vector_limit,
                distance_threshold=distance_threshold,
                **filters,
            )
            lexical_rows = []
            if is_hybrid:
                lexical_rows = await store.search_lexical(
                    session,
                    query_text=query_text,
                    top_k=vector_limit,
                    **filters,
                )
    except Exception as exc:  # noqa: BLE001 — DB/connection failure.
        log.error(
            "rag_hybrid_search_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise RetrievalError("Vector store query failed.") from exc

    # One Candidate per unique chunk id; the dense branch owns the distance.
    by_id: dict[int, Candidate] = {}
    for row in vector_rows:
        by_id[row.id] = _candidate_from_row(row, distance=float(row.distance))
    for row in lexical_rows:
        if row.id not in by_id:
            by_id[row.id] = _candidate_from_row(row, distance=None)

    if is_hybrid:
        vector_ranking = [row.id for row in vector_rows]
        lexical_ranking = [row.id for row in lexical_rows]
        fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking], k=rrf_k)
        ordered = [by_id[cid] for cid, _score in fused]
    else:
        # search_filtered already returns ascending distance (best first).
        ordered = [by_id[row.id] for row in vector_rows]

    if rerank:
        reranker = reranker or get_reranker()
        if reranker is None:
            log.warning("reranker_unavailable_falling_back", reason="no_reranker")
            ranked = ordered[:top_k]
        else:
            ranked = await asyncio.to_thread(
                reranker.rerank, query_text, ordered, top_n=rerank_top_n
            )
    else:
        ranked = ordered[:top_k]

    log.info(
        "rag_retrieve_done",
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_rows),
        returned=len(ranked),
        candidates_evaluated=candidates_evaluated,
        retrieve_time_ms=int((time.perf_counter() - started) * 1000),
    )
    return ranked, candidates_evaluated


async def retrieve(
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: SearchMode,
    rerank: bool,
    top_k: int | None = None,
    recall_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    distance_threshold: float | None = None,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker=None,
) -> RetrievalResult:
    """Hybrid/reranked retrieval projected onto the Session 9 :class:`RetrievalResult`.

    Drop-in for ``retriever.search_chunks`` with two extra switches; ``low_confidence``
    keeps the soft-fail contract (True iff nothing was retrieved)."""
    candidates, candidates_evaluated = await retrieve_ranked(
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=top_k,
        recall_k=recall_k,
        rerank_top_n=rerank_top_n,
        rrf_k=rrf_k,
        distance_threshold=distance_threshold,
        sectors=sectors,
        project_year_min=project_year_min,
        project_year_max=project_year_max,
        chunk_types=chunk_types,
        reranker=reranker,
    )
    return RetrievalResult(
        chunks=[c.to_retrieved_chunk() for c in candidates],
        low_confidence=not candidates,
        candidates_evaluated=candidates_evaluated,
    )
