"""Unit tests for Reciprocal Rank Fusion (Session 10).

Pure function, no I/O: we assert the position-based fusion math, the canonical
``k=60`` smoothing, agreement boosting, deterministic tie-breaking and edge
cases (empty inputs, duplicates, missing-from-a-branch)."""

from __future__ import annotations

import pytest

from app.generation.rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [key for key, _ in fused] == ["a", "b", "c"]


def test_scores_follow_the_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_default_k_is_60() -> None:
    assert DEFAULT_RRF_K == 60
    fused = dict(reciprocal_rank_fusion([["x"]]))
    assert fused["x"] == pytest.approx(1 / 61)


def test_agreement_across_branches_outranks_a_single_first_place() -> None:
    # "b" is 2nd in both branches; "a" is 1st in one and absent from the other.
    vector = ["a", "b", "c"]
    lexical = ["d", "b", "e"]
    fused = reciprocal_rank_fusion([vector, lexical], k=60)
    ranked = [key for key, _ in fused]
    # b: 1/62 + 1/62 = 0.03226 ; a: 1/61 = 0.01639 → b wins on agreement.
    assert ranked[0] == "b"


def test_document_missing_from_a_branch_gets_no_penalty() -> None:
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
    # Each appears once at rank 1 in its own branch.
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_duplicates_within_one_ranking_count_once_at_best_rank() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)  # rank 1, the later dup ignored
    assert fused["b"] == pytest.approx(1 / 63)  # still at its physical rank 3


def test_ties_break_by_first_seen_order_deterministically() -> None:
    # Two singleton branches → equal scores; "a" was seen before "b".
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert [key for key, _ in fused] == ["a", "b"]


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_non_positive_k_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
