#!/usr/bin/env python3
"""Warm up / smoke-test the cross-encoder reranker (Session 10).

The reranker loads its torch weights lazily on the first ``rerank`` call. That
first load downloads the model and can take several seconds — latency you do not
want a real request to pay. This script forces the load and runs one tiny
``(query, document)`` rescoring so you can:

* pre-download the weights into the image/volume,
* confirm ``sentence-transformers`` + torch import cleanly on this host,
* eyeball that a relevant document outranks an irrelevant one.

Usage::

    docker compose run --rm estimator python scripts/verify_reranker.py
    # or, locally:
    uv run python scripts/verify_reranker.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.generation.rag.retrieval.reranker import CrossEncoderReranker  # noqa: E402


@dataclass
class _Doc:
    id: int
    content: str


def main() -> int:
    settings = get_settings()
    reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)

    print(f"Loading reranker model: {reranker.model_name} ...")
    t0 = time.perf_counter()
    reranker.ensure_loaded()
    print(f"Loaded in {time.perf_counter() - t0:.1f} s.")

    query = "OAuth 2.0 authentication backend with JWT sessions for a bank"
    docs = [
        _Doc(1, "OAuth 2.0 authorization code flow with JWT session management and rate limiting."),
        _Doc(2, "Daily delivery slot booking with optimistic locking to prevent overbooking."),
        _Doc(3, "Strong customer authentication and consent management for PSD2 open banking."),
    ]

    ranked = reranker.rerank(query, docs, top_n=len(docs))
    print("\nRerank result (best first):")
    for position, doc in enumerate(ranked, start=1):
        print(f"  {position}. [doc {doc.id}] {doc.content}")

    if ranked and ranked[0].id in (1, 3):
        print("\nOK: an authentication document ranked first.")
        return 0
    print("\nWARNING: unexpected ordering — inspect the model output above.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
