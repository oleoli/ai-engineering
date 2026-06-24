"""Cross-encoder reranker — the fine half of recall-then-rerank (Session 10).

A bi-encoder (the embedding model) encodes the query and each document
*separately*: fast, indexable, but it never lets the two texts attend to each
other, so it misses fine-grained relevance. A **cross-encoder** feeds the pair
``(query, document)`` through the model *together* and attends across both — far
more precise, but far too slow to run over the whole corpus. So the pipeline
does cheap broad recall first (dense/hybrid, ``RETRIEVAL_RECALL_TOP_K``) and lets
the cross-encoder rescas only that shortlist down to ``RERANK_TOP_N``.

Operational notes:

* **Lazy load.** ``sentence-transformers`` pulls in torch and downloads the
  model weights on first use. We do NOT pay that at import/startup — the model
  loads on the first :meth:`rerank` call, guarded by a lock so concurrent
  requests load it exactly once. ``scripts/verify_reranker.py`` forces the load
  for warmup.
* **CPU + multilingual.** The default model (ES+EN) is small enough for CPU.
  ``predict`` is synchronous and CPU-bound, so async callers push it to a thread.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, Sequence

import structlog

log = structlog.get_logger()

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class _Rerankable(Protocol):
    """Anything carrying an ``id`` and the ``content`` the cross-encoder scores."""

    id: int
    content: str


class CrossEncoderReranker:
    """Rescas (query, document) pairs with a lazily-loaded cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the cross-encoder weights once (double-checked locking)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder

            t0 = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            log.info(
                "reranker_loaded",
                model=self._model_name,
                load_time_ms=int((time.perf_counter() - t0) * 1000),
            )

    def rerank(
        self,
        query: str,
        candidates: Sequence[_Rerankable],
        *,
        top_n: int,
    ) -> list[_Rerankable]:
        """Return the ``top_n`` candidates most relevant to ``query``, best first.

        Loads the model on first call. An empty candidate list short-circuits
        (no model load), so a soft-failed recall never triggers a torch download.
        """
        if not candidates:
            return []

        self.ensure_loaded()
        pairs = [(query, candidate.content) for candidate in candidates]

        t0 = time.perf_counter()
        scores = self._model.predict(pairs)  # type: ignore[union-attr]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        log.info(
            "reranker_done",
            candidates=len(candidates),
            top_n=top_n,
            rerank_time_ms=int((time.perf_counter() - t0) * 1000),
        )
        return [candidate for candidate, _ in ranked[:top_n]]
