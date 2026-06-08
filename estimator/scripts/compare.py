#!/usr/bin/env python3
"""Semantic search sanity check — five representative queries against /embeddings/search.

Invokes ``POST /embeddings/search`` with five queries that exercise the ingested
corpus from different angles (direct match, semantic reformulation, out-of-domain,
ambiguous, and highly specific). Prints the top-k results per query.

Requires the estimator service running with the corpus already ingested via
``POST /embeddings/ingest``.

Usage::

    # outside the container (from the estimator/ dir):
    uv run python scripts/compare.py

    # inside the container:
    docker compose exec estimator python scripts/compare.py

    # override base URL or result count:
    uv run python scripts/compare.py --base-url http://localhost:8000 --k 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

# (label, query) — each label describes the retrieval angle being tested.
REPRESENTATIVE_QUERIES: list[tuple[str, str]] = [
    (
        "direct",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "semantic",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "out_of_domain",
        "mobile application for restaurant reservations",
    ),
    (
        "ambiguous",
        "integration with external system",
    ),
    (
        "specific",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

CONTENT_PREVIEW_LEN = 120


def _preview(content: str, max_len: int = CONTENT_PREVIEW_LEN) -> str:
    """First ~max_len characters on a single line."""
    collapsed = re.sub(r"\s+", " ", content).strip()
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1] + "…"


def _print_results(label: str, query: str, results: list[dict]) -> None:
    print(f"\n=== [{label}] {query} ===")
    if not results:
        print("  (no results)")
        return
    for rank, item in enumerate(results, 1):
        chunk_id = item.get("chunk_id", "?")
        distance = item.get("distance", 0.0)
        chunk_type = item.get("chunk_type", "?")
        content = _preview(item.get("content", ""))
        print(
            f"  {rank}. chunk_id={chunk_id}  dist={distance:.4f}  "
            f"type={chunk_type}  {content}"
        )


def run_search(base_url: str, query: str, k: int) -> tuple[int, dict | str]:
    """POST to /embeddings/search; return (status_code, parsed_json_or_error_text)."""
    url = f"{base_url.rstrip('/')}/embeddings/search"
    try:
        response = httpx.post(url, json={"query": query, "k": k}, timeout=60.0)
    except httpx.ConnectError:
        return 0, (
            f"Cannot connect to {url}. "
            "Start the service (docker compose up / uvicorn) and ingest the corpus first."
        )
    except httpx.HTTPError as exc:
        return 0, f"HTTP error: {exc}"

    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run five representative semantic-search queries against /embeddings/search."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Estimator base URL (default: ESTIMATOR_API_BASE_URL or http://localhost:8000).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="N",
        help="Number of results per query (default: 5).",
    )
    args = parser.parse_args()

    settings = get_settings()
    base_url = (
        args.base_url
        or os.getenv("ESTIMATOR_API_BASE_URL")
        or settings.ESTIMATOR_API_BASE_URL
    )

    print(f"Target: {base_url.rstrip('/')}/embeddings/search  (k={args.k})")
    print(f"Queries: {len(REPRESENTATIVE_QUERIES)}")

    exit_code = 0
    for label, query in REPRESENTATIVE_QUERIES:
        status, body = run_search(base_url, query, args.k)
        if status == 0:
            print(f"\n=== [{label}] {query} ===", file=sys.stderr)
            print(f"ERROR: {body}", file=sys.stderr)
            exit_code = 1
            break
        if status != 200:
            print(f"\n=== [{label}] {query} ===", file=sys.stderr)
            detail = body if isinstance(body, str) else body.get("detail", body)
            print(f"ERROR: HTTP {status} — {detail}", file=sys.stderr)
            exit_code = 1
            continue

        results = body.get("results", []) if isinstance(body, dict) else []
        search_ms = body.get("search_time_ms", "?") if isinstance(body, dict) else "?"
        _print_results(label, query, results)
        print(f"  ({search_ms} ms)")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
