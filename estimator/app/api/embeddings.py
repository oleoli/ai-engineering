"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker -> embedder -> response assembly and maps
failures to status codes. No business logic lives here.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.foundation.persistence.database import get_async_session
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.generation.rag.embedding.embedder import EMBEDDING_DIM, OpenAIEmbedder
from app.generation.rag.schemas import IngestRequest, IngestResponse
from app.generation.rag.store.models import ChunkRow, DocumentRow

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_async_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    ) -> IngestResponse | JSONResponse:
    """Chunk a budget, embed every chunk, persist in one transaction, return metrics."""
    if embedder is None:
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    t0 = time.perf_counter()

    existing_id = await session.scalar(
        select(DocumentRow.id).where(DocumentRow.source_path == request.source_path)
    )
    if existing_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Document already ingested",
                "document_id": existing_id,
            },
        )

    document = DocumentRow(
        source_path=request.source_path,
        document_type=request.document_type,
    )
    session.add(document)
    await session.flush()

    chunks = chunker.chunk([request.content])
    log.info(
        "embeddings_ingest_received",
        source_path=request.source_path,
        document_type=request.document_type,
        total_chunks=len(chunks),
    )

    try:
        embedded = await run_in_threadpool(embedder.embed_many, chunks)
    except Exception as exc:  # noqa: BLE001 — any embedding-API failure becomes a 500.
        await session.rollback()
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc

    chunk_rows = [
        ChunkRow(
            document_id=document.id,
            chunk_type=chunker.strategy_name,
            content=chunk.text,
            embedding=chunk.embedding,
            chunk_metadata=chunk.metadata,
        )
        for chunk in embedded
    ]
    session.add_all(chunk_rows)
    await session.commit()

    embedding_dimension = len(embedded[0].embedding) if embedded else EMBEDDING_DIM
    ingestion_time_ms = round((time.perf_counter() - t0) * 1000)

    response = IngestResponse(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=embedding_dimension,
        ingestion_time_ms=ingestion_time_ms,
    )
    log.info("embeddings_ingest_done", **response.model_dump())
    return response


@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets and compare them.

    Returns per-strategy corpus stats and, if queries are given, the top-k
    chunks each strategy retrieves. Nothing is persisted (Session 8 territory).
    """
    if embedder is None:
        log.error("embeddings_compare_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc.args[0]}") from exc
    except RuntimeError as exc:
        # A strategy needs an API key that is not configured.
        log.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:  # noqa: BLE001 — any chunker/embedding failure becomes a 500.
        log.error(
            "embeddings_compare_failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run chunking comparison.") from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)
