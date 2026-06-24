"""Reciprocal Rank Fusion (RRF) — combine rankings by position, not by score.

The dense branch scores by cosine *distance* (lower is better) and the lexical
branch by ``ts_rank_cd`` (higher is better): two incomparable scales. Normalising
them against each other is fragile and corpus-dependent. RRF sidesteps the whole
problem by throwing the scores away and trusting only each branch's *order*:

    score(d) = Σ_branches 1 / (k + rank_branch(d))

where ``rank`` is 1-based and ``k`` (``RRF_K``, default 60 — Cormack et al. 2009)
is a smoothing constant that damps the weight of any single branch's top
positions, so a document must do well across branches to rise, and neither
branch can dominate on its own. A document missing from a branch simply
contributes nothing from that branch (no penalty term).

Pure and synchronous: no I/O, trivially unit-testable.
"""

from __future__ import annotations

from typing import Hashable, Sequence, TypeVar

K = TypeVar("K", bound=Hashable)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[K, float]]:
    """Fuse several ranked lists of keys into one, by Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        One sequence per branch, each an ordered list of keys (best first).
        Keys must be hashable (chunk ids in our case). Duplicates within a
        single ranking are ignored after their first (best) occurrence.
    k:
        RRF smoothing constant (``RRF_K``). Must be positive.

    Returns
    -------
    list[tuple[K, float]]
        ``(key, fused_score)`` pairs sorted by descending score. Ties are broken
        deterministically by the key's first-seen order across the input
        rankings, so the fusion is stable and reproducible.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[K, float] = {}
    first_seen: dict[K, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_ranking: set[K] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in first_seen:
                first_seen[key] = order
                order += 1

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
