#!/usr/bin/env python3
"""Ingest the sample budget corpus into Postgres via POST /embeddings/ingest.

Reads ``data/budgets_sample.json`` (an array of budgets) and sends one ingest
request per budget. Each document is keyed by ``source_path`` so re-running
skips already-ingested budgets (HTTP 409).

Requires the estimator service running with migrations applied.

Usage::

    # outside the container (from the estimator/ dir):
    uv run python scripts/ingest_corpus.py

    # inside the container:
    docker compose exec estimator python scripts/ingest_corpus.py

    # custom file or base URL:
    uv run python scripts/ingest_corpus.py --file data/budgets_sample.json --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

DEFAULT_BUDGETS_FILE = ROOT / "data" / "budgets_sample.json"
DOCUMENT_TYPE = "historical_budget"
INGEST_TIMEOUT_S = 120.0


def _source_path(budgets_file: Path, budget_id: str) -> str:
    """Stable idempotency key: relative path + budget id."""
    try:
        rel = budgets_file.relative_to(ROOT)
    except ValueError:
        rel = budgets_file
    return f"{rel.as_posix()}#{budget_id}"


def ingest_budget(
    client: httpx.Client,
    base_url: str,
    budgets_file: Path,
    budget: dict,
) -> tuple[str, int, dict | str]:
    """POST one budget; return (budget_id, status_code, body_or_error)."""
    budget_id = budget["budget_id"]
    payload = {
        "source_path": _source_path(budgets_file, budget_id),
        "document_type": DOCUMENT_TYPE,
        "content": budget,
    }
    url = f"{base_url.rstrip('/')}/embeddings/ingest"
    try:
        response = client.post(url, json=payload, timeout=INGEST_TIMEOUT_S)
    except httpx.ConnectError:
        return budget_id, 0, (
            f"Cannot connect to {url}. "
            "Start the service (docker compose up / uvicorn) and retry."
        )
    except httpx.HTTPError as exc:
        return budget_id, 0, f"HTTP error: {exc}"

    try:
        body: dict | str = response.json()
    except ValueError:
        body = response.text
    return budget_id, response.status_code, body


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest all budgets from budgets_sample.json via /embeddings/ingest."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_BUDGETS_FILE,
        help=f"JSON array of budgets (default: {DEFAULT_BUDGETS_FILE.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Estimator base URL (default: ESTIMATOR_API_BASE_URL or http://localhost:8000).",
    )
    args = parser.parse_args()

    budgets_file: Path = args.file
    if not budgets_file.is_absolute():
        budgets_file = ROOT / budgets_file
    if not budgets_file.is_file():
        print(f"ERROR: file not found: {budgets_file}", file=sys.stderr)
        return 1

    settings = get_settings()
    base_url = (
        args.base_url
        or os.getenv("ESTIMATOR_API_BASE_URL")
        or settings.ESTIMATOR_API_BASE_URL
    )

    raw = json.loads(budgets_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        print(f"ERROR: expected a non-empty JSON array in {budgets_file}", file=sys.stderr)
        return 1

    print(f"Source: {budgets_file}")
    print(f"Target: {base_url.rstrip('/')}/embeddings/ingest")
    print(f"Budgets: {len(raw)}\n")

    ok = skip = fail = 0
    total_chunks = 0

    with httpx.Client() as client:
        for budget in raw:
            budget_id, status, body = ingest_budget(client, base_url, budgets_file, budget)
            if status == 0:
                print(f"ERROR [{budget_id}]: {body}", file=sys.stderr)
                return 1
            if status == 200 and isinstance(body, dict):
                chunks = body.get("chunks_created", 0)
                doc_id = body.get("document_id", "?")
                ms = body.get("ingestion_time_ms", "?")
                total_chunks += chunks
                print(f"OK   {budget_id}  document_id={doc_id}  chunks={chunks}  ({ms} ms)")
                ok += 1
            elif status == 409 and isinstance(body, dict):
                doc_id = body.get("document_id", "?")
                print(f"SKIP {budget_id}  already ingested (document_id={doc_id})")
                skip += 1
            else:
                detail = body if isinstance(body, str) else body.get("detail", body)
                print(f"FAIL {budget_id}  HTTP {status}  {detail}", file=sys.stderr)
                fail += 1

    print(f"\nSummary: {ok} ingested, {skip} skipped, {fail} failed, {total_chunks} new chunks")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
