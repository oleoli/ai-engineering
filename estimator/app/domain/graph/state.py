"""Typed state for the Session 13 estimation graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Component(TypedDict):
    """A functional component derived from the transcript brief."""

    name: str
    category: str


class BudgetMatch(TypedDict):
    """One historical budget reference found for a component."""

    component: str
    reference_budget_id: str
    amount: float


class EstimationState(TypedDict, total=False):
    """Graph state threaded through every node.

    ``budget_matches`` and ``errors`` use reducers so parallel ``search_budgets``
    branches (future Send API) can append without clobbering prior results.
    """

    transcript: str
    estimation_id: str
    query: dict | None
    requirements: list[str]
    components: list[Component]
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    retrieved_chunks: list[dict]
    estimate: dict | None
    status: str | None  # "validated" | "needs_review"
    errors: Annotated[list[str], operator.add]
