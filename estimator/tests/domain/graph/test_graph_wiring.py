"""Network-free wiring tests for the Session 13 estimation graph."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.domain.graph.build import build_graph
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    SourceReference,
    TaskItem,
    WorkModule,
)


@pytest.mark.asyncio
async def test_graph_sequential_wiring_produces_validated_status() -> None:
    """The five nodes run in order and surface estimate + workflow status."""
    query = EstimationQuery(function="B2B payments platform", technologies=["Stripe"])
    structure = Estimate(
        confidence="medium",
        reasoning="structure only",
        modules=[],
    )
    insufficient = Estimate(
        confidence="insufficient",
        reasoning="no budgets",
        insufficient_context_explanation="No historical budgets were found for any classified component.",
        modules=[],
    )

    with (
        patch("app.domain.graph.nodes.reformulate_query", new=AsyncMock(return_value=query)),
        patch("app.domain.graph.nodes.generate_structure", new=AsyncMock(return_value=structure)),
        patch(
            "app.domain.graph.nodes.rag_generate_estimate",
            new=AsyncMock(return_value=insufficient),
        ),
    ):
        graph = build_graph(MemorySaver())
        result = await graph.ainvoke(
            {"transcript": "x" * 120, "estimation_id": "test-thread"},
            {"configurable": {"thread_id": "test-thread"}},
        )

    assert result["status"] == "validated"
    assert result["estimate"]["confidence"] == "insufficient"
    assert result["budget_matches"] == []


@pytest.mark.asyncio
async def test_graph_marks_needs_review_on_dangling_citations() -> None:
    query = EstimationQuery(function="ERP integration")
    structure = Estimate(
        confidence="medium",
        reasoning="structure",
        modules=[
            WorkModule(
                name="Integration",
                description="ERP",
                tasks=[TaskItem(name="Sync", description="data sync", grounded=False, sources=[])],
            )
        ],
    )
    grounded = Estimate(
        total_engineer_days=10,
        confidence="high",
        reasoning="grounded",
        modules=[
            WorkModule(
                name="Integration",
                description="ERP",
                tasks=[
                    TaskItem(
                        name="Sync",
                        description="data sync",
                        engineer_days=10,
                        grounded=True,
                        sources=[
                            SourceReference(
                                chunk_id="missing",
                                document_id="d1",
                                evidence="fabricated",
                            )
                        ],
                    )
                ],
            )
        ],
    )

    chunk = {
        "id": 99,
        "content": "task",
        "sector": "industrial",
        "project_year": 2023,
        "chunk_type": "historical_task",
        "distance": 0.1,
        "budget_id": "BUD-099",
        "collection": "budget",
        "source_id": "99",
        "estimated_hours": 40,
        "document_date": None,
        "relevance_score": None,
    }

    from app.generation.rag.schemas import RetrievedChunk

    retrieval_chunk = RetrievedChunk.model_validate(chunk)

    class _RetrievalResult:
        chunks = [retrieval_chunk]

    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = [0.1, 0.2]

    with (
        patch("app.domain.graph.nodes.reformulate_query", new=AsyncMock(return_value=query)),
        patch("app.domain.graph.nodes.generate_structure", new=AsyncMock(return_value=structure)),
        patch(
            "app.domain.graph.nodes.retrieve",
            new=AsyncMock(return_value=_RetrievalResult()),
        ),
        patch("app.domain.graph.nodes.rag_generate_estimate", new=AsyncMock(return_value=grounded)),
        patch("app.dependencies.get_embedder", return_value=mock_embedder),
    ):
        graph = build_graph(MemorySaver())
        result = await graph.ainvoke(
            {"transcript": "y" * 120, "estimation_id": "review-thread"},
            {"configurable": {"thread_id": "review-thread"}},
        )

    assert result["status"] == "needs_review"
    assert result["estimate"]["confidence"] == "low"
    assert result["budget_matches"]
