#!/usr/bin/env python3
"""Session 13 — run the explicit LangGraph estimation pipeline over a transcript.

Observability: **Logfire** (one span per graph node via ``logfire.span`` in each
node). Configure ``LOGFIRE_TOKEN`` in ``.env`` to export traces to Logfire Cloud;
without a token spans are collected locally only.

The graph is checkpointed on the project's Postgres (pgvector instance). The
checkpointer creates its own tables on first ``setup()`` and coexists with the
embedding tables.

    # Offline-friendly: needs Postgres up (docker compose) but no LLM if you
    # monkeypatch — for a real run you need OPENAI_API_KEY + ingested corpus:
    uv run python scripts/run_graph_s13.py \\
        exercises/session-12/sample_transcript_complex.txt

    # Inside the estimator container (recommended for the live session):
    docker compose exec estimator python scripts/run_graph_s13.py \\
        exercises/session-12/sample_transcript_complex.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import logfire
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.domain.graph.build import build_graph  # noqa: E402
from app.domain.graph.schemas import GraphEstimateResponse  # noqa: E402
from app.foundation.persistence.database import langgraph_conn_string  # noqa: E402
from app.generation.rag.schemas import Estimate  # noqa: E402


def _configure_logfire() -> None:
    settings = get_settings()
    configure_kwargs: dict = {
        "send_to_logfire": "if-token-present",
        "service_name": settings.LOGFIRE_SERVICE_NAME,
    }
    if settings.LOGFIRE_TOKEN:
        configure_kwargs["token"] = settings.LOGFIRE_TOKEN
    logfire.configure(**configure_kwargs)
    logfire.instrument_asyncpg()
    logfire.instrument_httpx()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Session 13 estimation graph.")
    parser.add_argument(
        "transcript",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "exercises" / "session-12" / "sample_transcript_complex.txt",
        help="Path to a transcript text file.",
    )
    parser.add_argument(
        "--estimation-id",
        default=None,
        help="Thread id for the Postgres checkpointer (defaults to a fresh uuid).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON response.",
    )
    return parser.parse_args()


async def _run(transcript_path: Path, estimation_id: str) -> GraphEstimateResponse:
    transcript = transcript_path.read_text(encoding="utf-8")
    if len(transcript.strip()) < 100:
        raise SystemExit(f"Transcript too short (<100 chars): {transcript_path}")

    with logfire.span("graph_run", estimation_id=estimation_id, transcript=str(transcript_path)):
        async with AsyncPostgresSaver.from_conn_string(langgraph_conn_string()) as checkpointer:
            await checkpointer.setup()
            graph = build_graph(checkpointer)
            result = await graph.ainvoke(
                {"transcript": transcript, "estimation_id": estimation_id},
                {"configurable": {"thread_id": estimation_id}},
            )

    raw_estimate = result.get("estimate")
    if not raw_estimate:
        raise SystemExit("Graph finished without an estimate.")

    status = result.get("status") or "needs_review"
    if status not in ("validated", "needs_review"):
        status = "needs_review"

    return GraphEstimateResponse(
        estimate=Estimate.model_validate(raw_estimate),
        status=status,  # type: ignore[arg-type]
    )


def main() -> None:
    args = _parse_args()
    estimation_id = args.estimation_id or str(uuid4())
    _configure_logfire()

    response = asyncio.run(_run(args.transcript.resolve(), estimation_id))

    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nstatus={response.status}  estimation_id={estimation_id}")

    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
