"""``POST /v1/estimate/graph/from-transcript`` — LangGraph estimation (S13).

Same transport contract as the Session 9 endpoint (auth, rate limit, 502 on
pipeline failure) but the response wraps the :class:`Estimate` with an explicit
workflow ``status`` (``validated`` | ``needs_review``). The graph state is
checkpointed per ``estimation_id`` (the idempotency key or a fresh uuid).
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.domain.graph.schemas import GraphEstimateResponse
from app.generation.rag.errors import RagError
from app.generation.rag.schemas import Estimate, EstimateRequest

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate/graph", tags=["estimate-graph"])


@router.post(
    "/from-transcript",
    response_model=GraphEstimateResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def from_transcript(request: Request, payload: EstimateRequest) -> GraphEstimateResponse:
    """Run the explicit LangGraph pipeline over a raw transcript."""
    estimation_id = payload.idempotency_key or str(uuid4())
    graph = request.app.state.estimation_graph
    try:
        result = await graph.ainvoke(
            {"transcript": payload.transcript, "estimation_id": estimation_id},
            {"configurable": {"thread_id": estimation_id}},
        )
    except RagError as exc:
        log.error(
            "graph_estimate_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=502, detail="Failed to produce an estimate.") from exc
    except Exception as exc:  # noqa: BLE001 — unexpected graph/node failure
        log.error(
            "graph_estimate_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=502, detail="Failed to produce an estimate.") from exc

    raw_estimate = result.get("estimate")
    if not raw_estimate:
        raise HTTPException(
            status_code=502,
            detail="Graph finished without an estimate.",
        )

    status = result.get("status") or "needs_review"
    if status not in ("validated", "needs_review"):
        status = "needs_review"

    return GraphEstimateResponse(
        estimate=Estimate.model_validate(raw_estimate),
        status=status,  # type: ignore[arg-type]
    )
