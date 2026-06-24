"""Session 10 — hybrid retrieval (dense + lexical) and cross-encoder reranking.

Public surface:

* :func:`reciprocal_rank_fusion` — pure RRF fusion of N rankings.
* :class:`CrossEncoderReranker` — lazy cross-encoder for recall-then-rerank.
* :func:`retrieve` / :func:`retrieve_ranked` — the single retrieval entry point
  composing the four (search_mode × rerank) configurations.
"""

from __future__ import annotations

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.pipeline import Candidate, retrieve, retrieve_ranked
from app.generation.rag.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "Candidate",
    "CrossEncoderReranker",
    "reciprocal_rank_fusion",
    "retrieve",
    "retrieve_ranked",
]
