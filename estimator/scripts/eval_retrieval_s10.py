#!/usr/bin/env python3
"""Session 10 retrieval eval: precision@5 + latency across the four configs.

Runs the hand-annotated golden set (``evals/golden_retrieval.json``: five
project-to-estimate descriptions, each labelled with the genuinely relevant
historical budgets) through the four (search_mode x rerank) configurations and
reports **precision@5** and **query latency** in a comparison table:

    Config   Search    Reranking
    A        vector    no
    B        hybrid    no
    C        vector    yes
    D        hybrid    yes

The four configs are pinned explicitly (not read from the runtime store), so the
run is reproducible regardless of the live toggles. Query embeddings are computed
once up front and EXCLUDED from the timings — an embedding round-trip is an
OpenAI cost the retrieval config cannot influence. The cross-encoder is warmed up
before timing so the first reranked query is not charged the one-off weight load.

A permissive distance threshold is used on purpose: this measures *ranking
quality*, not the production soft-fail floor.

Prerequisites: a live stack with the base corpus ingested
(``data/budgets_sample.json``), e.g.::

    docker compose up -d
    docker compose run --rm estimator python scripts/query_examples.py     # ingest
    docker compose run --rm estimator python scripts/eval_retrieval_s10.py  # eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
# ROOT (project root) so ``app`` is importable; SCRIPTS_DIR so ``s08_common`` is
# (the latter is sys.path[0] when run as a script, added explicitly for safety).
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from s08_common import Stopwatch, format_table, require_embedder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dependencies import get_reranker  # noqa: E402
from app.generation.rag.retrieval import retrieve_ranked  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# Permissive threshold: we want to score the ranking, not trip the soft-fail.
PERMISSIVE_DISTANCE_THRESHOLD = 2.0

# (label, search_mode, rerank) — the four configurations under test.
CONFIGS: list[tuple[str, str, bool]] = [
    ("A", "vector", False),
    ("B", "hybrid", False),
    ("C", "vector", True),
    ("D", "hybrid", True),
]


def load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        print(f"ERROR: golden set not found at {GOLDEN_PATH}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def precision_at_k(budget_ids: list[str | None], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks whose budget is relevant."""
    top = budget_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for bid in top if bid in relevant)
    return hits / k


async def run_config(
    *,
    search_mode: str,
    rerank: bool,
    embedded_queries: list[tuple[dict, list[float]]],
    k: int,
    reranker,
) -> tuple[list[float], list[float]]:
    """Return (precision@k per query, latency_ms per query) for one config."""
    settings = get_settings()
    precisions: list[float] = []
    latencies: list[float] = []

    for query, embedding in embedded_queries:
        relevant = set(query["relevant_budget_ids"])
        with Stopwatch() as sw:
            candidates, _evaluated = await retrieve_ranked(
                query_embedding=embedding,
                query_text=query["query"],
                search_mode=search_mode,
                rerank=rerank,
                top_k=k,
                recall_k=settings.RETRIEVAL_RECALL_TOP_K,
                rerank_top_n=k,
                rrf_k=settings.RRF_K,
                distance_threshold=PERMISSIVE_DISTANCE_THRESHOLD,
                reranker=reranker,
            )
        budget_ids = [c.budget_id for c in candidates]
        precisions.append(precision_at_k(budget_ids, relevant, k))
        latencies.append(sw.elapsed_ms)

    return precisions, latencies


async def main() -> int:
    golden = load_golden()
    queries = golden["queries"]
    k = int(golden.get("k", 5))

    embedder = require_embedder()
    print(f"Embedding {len(queries)} golden queries (excluded from timings)...")
    embedded_queries = [(q, embedder.embed_one(q["query"])) for q in queries]

    # Warm the cross-encoder up front so config C's first query is not charged
    # the one-off torch weight load.
    reranker = get_reranker()
    if any(rerank for _, _, rerank in CONFIGS) and reranker is not None:
        print(f"Warming up reranker ({reranker.model_name})...")
        reranker.ensure_loaded()

    summary_rows: list[list[str]] = []
    print()
    for label, search_mode, rerank in CONFIGS:
        precisions, latencies = await run_config(
            search_mode=search_mode,
            rerank=rerank,
            embedded_queries=embedded_queries,
            k=k,
            reranker=reranker,
        )
        mean_p = sum(precisions) / len(precisions)
        mean_lat = sum(latencies) / len(latencies)

        print(f"[Config {label}] search={search_mode:<6} rerank={'yes' if rerank else 'no':<3}")
        for query, p, lat in zip(queries, precisions, latencies):
            print(f"    {query['id']:<32}  P@{k}={p:.2f}  {lat:7.1f} ms")
        print(f"    {'mean':<32}  P@{k}={mean_p:.2f}  {mean_lat:7.1f} ms\n")

        summary_rows.append(
            [
                label,
                search_mode,
                "yes" if rerank else "no",
                f"{mean_p:.2f}",
                f"{mean_lat:.1f}",
            ]
        )

    print("=" * 60)
    print(f"Comparative summary (golden set: {len(queries)} queries, k={k})\n")
    print(
        format_table(
            ["Config", "Search", "Reranking", f"precision@{k}", "latency_ms"],
            summary_rows,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
