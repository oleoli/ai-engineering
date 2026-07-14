"""HTTP response models for the LangGraph estimation endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.generation.rag.schemas import Estimate

GraphEstimateStatus = Literal["validated", "needs_review"]


class GraphEstimateResponse(BaseModel):
    """Wrapper keeping the external contract: structured estimate + workflow status."""

    estimate: Estimate
    status: GraphEstimateStatus = Field(
        description="Whether the estimate passed validation or needs human review."
    )
