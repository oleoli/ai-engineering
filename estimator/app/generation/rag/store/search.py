"""Semantic search over persisted chunk embeddings."""

from __future__ import annotations

from sqlalchemy import Row, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.store.models import ChunkRow


async def search_similar_chunks(
    session: AsyncSession,
    query_vector: list[float],
    k: int,
) -> list[Row]:
    """Return the top-k chunks closest to ``query_vector`` by cosine distance."""
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.chunk_metadata,
            distance,
        )
        .where(ChunkRow.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    result = await session.execute(stmt)
    return list(result.all())
