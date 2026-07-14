"""LangGraph node functions for the Session 13 estimation pipeline.

Each node is a pure-ish async function: it receives the graph state and returns a
partial update. Business logic is delegated to the existing RAG layer (Sessions
9–11); this module only orchestrates and maps shapes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import logfire
import structlog

from app.config import get_settings
from app.domain.graph.state import BudgetMatch, Component, EstimationState
from app.generation.rag.context_assembler import build_context_block, truncate_to_token_budget
from app.generation.rag.estimator import generate_estimate as rag_generate_estimate
from app.generation.rag.estimator import generate_structure
from app.generation.rag.query_reformulator import reformulate_query
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.schemas import Estimate, EstimationQuery, RetrievedChunk
from app.generation.rag.validation import check_coherence, verify_citations_for_chunks

log = structlog.get_logger()

_KNOWN_SECTORS = {
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "media",
    "government",
}


def _requirements_from_query(query: EstimationQuery) -> list[str]:
    """Flatten the structured brief into a list of requirement strings."""
    items: list[str] = []
    if query.function.strip():
        items.append(query.function.strip())
    items.extend(query.technologies)
    items.extend(query.regulations)
    items.extend(query.constraints)
    return items


def _component_search_text(component: Component) -> str:
    """Compose a short retrieval query for one classified component."""
    category = component["category"].strip()
    if category:
        return f"{component['name']}: {category}"
    return component["name"]


def _insufficient_estimate(explanation: str) -> Estimate:
    return Estimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="Retrieval did not surface enough relevant historical budgets.",
        insufficient_context_explanation=explanation,
    )


def _chunks_from_state(raw_chunks: list[dict]) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(chunk) for chunk in raw_chunks]


async def extract_requirements(state: EstimationState) -> dict[str, Any]:
    """Distil the transcript into a structured brief and a requirements list."""
    with logfire.span("node: extract_requirements"):
        query = await reformulate_query(state["transcript"])
        requirements = _requirements_from_query(query)
        log.info(
            "graph_extract_requirements",
            estimation_id=state.get("estimation_id"),
            requirement_count=len(requirements),
        )
        return {
            "query": query.model_dump(),
            "requirements": requirements,
        }


async def classify_components(state: EstimationState) -> dict[str, Any]:
    """Group requirements into functional components (module tree)."""
    with logfire.span("node: classify_components"):
        query = EstimationQuery.model_validate(state["query"])
        structure = await generate_structure(query)
        components: list[Component] = [
            {
                "name": module.name,
                "category": module.description or "software_module",
            }
            for module in structure.modules
        ]
        log.info(
            "graph_classify_components",
            estimation_id=state.get("estimation_id"),
            component_count=len(components),
        )
        return {"components": components}


async def search_budgets(state: EstimationState) -> dict[str, Any]:
    """Retrieve one historical budget reference per component (sequential for now)."""
    with logfire.span("node: search_budgets"):
        from app.dependencies import get_embedder, get_runtime_retrieval_config

        settings = get_settings()
        embedder = get_embedder()
        if embedder is None:
            return {
                "budget_matches": [],
                "retrieved_chunks": [],
                "errors": ["Embedding service is not available (no OPENAI_API_KEY)."],
            }

        query = EstimationQuery.model_validate(state["query"])
        sector = query.sector.lower().strip() if query.sector else None
        sectors = [sector] if sector in _KNOWN_SECTORS else None
        runtime = get_runtime_retrieval_config()
        search_mode = runtime.effective_search_mode()
        rerank = runtime.effective_rerank()

        matches: list[BudgetMatch] = []
        chunk_by_id: dict[str, dict] = {}
        errors: list[str] = []

        for component in state.get("components") or []:
            search_text = _component_search_text(component)
            query_embedding = await asyncio.to_thread(embedder.embed_one, search_text)
            result = await retrieve(
                query_embedding=query_embedding,
                query_text=search_text,
                search_mode=search_mode,
                rerank=rerank,
                collection=Collection.BUDGET,
                chunk_types=["historical_task"],
                top_k=settings.AGENT_SEARCH_TOP_K,
                recall_k=settings.RETRIEVAL_RECALL_TOP_K,
                rerank_top_n=settings.RERANK_TOP_N,
                distance_threshold=settings.AGENT_SEARCH_DISTANCE_THRESHOLD,
                rrf_k=settings.RRF_K,
                sectors=sectors,
            )
            if not result.chunks:
                errors.append(f"No budget match for component {component['name']!r}.")
                continue

            best = result.chunks[0]
            amount = float(best.estimated_hours or 0)
            matches.append(
                {
                    "component": component["name"],
                    "reference_budget_id": best.budget_id or str(best.id),
                    "amount": amount,
                }
            )
            for chunk in result.chunks:
                chunk_by_id[str(chunk.id)] = chunk.model_dump()

        log.info(
            "graph_search_budgets",
            estimation_id=state.get("estimation_id"),
            match_count=len(matches),
            chunk_count=len(chunk_by_id),
        )
        return {
            "budget_matches": matches,
            "retrieved_chunks": list(chunk_by_id.values()),
            "errors": errors,
        }


async def generate_estimate(state: EstimationState) -> dict[str, Any]:
    """Assemble retrieved budgets and produce a grounded estimate."""
    with logfire.span("node: generate_estimate"):
        query = EstimationQuery.model_validate(state["query"])
        raw_chunks = state.get("retrieved_chunks") or []
        if not raw_chunks:
            estimate = _insufficient_estimate(
                "No historical budgets were found for any classified component."
            )
            return {"estimate": estimate.model_dump()}

        settings = get_settings()
        from app.dependencies import get_token_encoder

        chunks = _chunks_from_state(raw_chunks)
        encoder = get_token_encoder()
        kept = truncate_to_token_budget(chunks, settings.MAX_CONTEXT_TOKENS, encoder)
        if not kept:
            estimate = _insufficient_estimate(
                "Retrieved budgets exceeded the context token budget with no whole chunks kept."
            )
            return {"estimate": estimate.model_dump()}

        context_block = build_context_block(kept)
        estimate = await rag_generate_estimate(context_block, query, include_hours=True)
        log.info(
            "graph_generate_estimate",
            estimation_id=state.get("estimation_id"),
            confidence=estimate.confidence,
            module_count=len(estimate.modules),
        )
        return {"estimate": estimate.model_dump()}


async def validate_and_consolidate(state: EstimationState) -> dict[str, Any]:
    """Verify citations and coherence, then fix the workflow status."""
    with logfire.span("node: validate_and_consolidate"):
        raw_estimate = state.get("estimate")
        if not raw_estimate:
            return {
                "status": "needs_review",
                "errors": ["No estimate was produced before validation."],
            }

        estimate = Estimate.model_validate(raw_estimate)
        chunks = _chunks_from_state(state.get("retrieved_chunks") or [])
        report = verify_citations_for_chunks(estimate, chunks)
        coherent = check_coherence(estimate)

        validation_errors: list[str] = []
        if report.has_dangling:
            validation_errors.append(f"Dangling citations: {report.dangling_citations}")
        if not coherent:
            validation_errors.append("Estimate fails the insufficient-context coherence rule.")

        status = "validated" if not validation_errors else "needs_review"
        if status == "needs_review" and estimate.confidence in ("high", "medium"):
            estimate = estimate.model_copy(update={"confidence": "low"})

        log.info(
            "graph_validate_and_consolidate",
            estimation_id=state.get("estimation_id"),
            status=status,
            dangling=report.has_dangling,
            coherent=coherent,
        )
        update: dict[str, Any] = {
            "estimate": estimate.model_dump(),
            "status": status,
        }
        if validation_errors:
            update["errors"] = validation_errors
        return update
