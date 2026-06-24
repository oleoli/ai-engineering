"""Async data-access layer for the vector store.

The store never opens or commits sessions: the caller (ingest service,
retriever) owns the ``AsyncSession`` so a whole ingest — duplicate check,
document row, chunk rows — fits in ONE transaction. A failure anywhere rolls
everything back and leaves no orphan ``documents`` row.
"""

from __future__ import annotations

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Integer, Row, Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import UserDefinedType

from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import ChunkRow, DocumentRow, EMBEDDING_DIMENSIONS

# The structural chunker emits one chunk per budget component; the vocabulary
# is queryable thanks to the index on ``chunk_type`` (live-session filters).
BUDGET_COMPONENT = "budget_component"

# Postgres text-search configuration. MUST match the regconfig of the generated
# ``content_tsv`` column (migration ``0003_session10_fts``) or the GIN index is
# silently ignored and the lexical branch falls back to a sequential scan.
FTS_REGCONFIG = "english"


class _TSQuery(UserDefinedType):
    """Minimal SQLAlchemy mapping for the Postgres ``tsquery`` type (cast target)."""

    cache_ok = True

    def get_col_spec(self, **kw):  # noqa: D401, ANN001, ANN003
        return "tsquery"


def _or_tsquery(query_text: str):
    """Build an **OR** full-text query from free text.

    ``plainto_tsquery`` ANDs every lexeme, which returns nothing for the long,
    descriptive queries this branch sees (no single chunk contains *all* the
    terms of a project description). We let Postgres normalize/stem the input
    with ``plainto_tsquery`` and then flip the ``&`` operators to ``|`` so a
    chunk matches on **any** salient term; ``ts_rank_cd`` still ranks chunks
    sharing more (and rarer) terms higher. The replace is safe because
    ``plainto_tsquery`` only ever emits ``&`` between quoted, sanitized lexemes.
    """
    plain = func.plainto_tsquery(FTS_REGCONFIG, query_text)
    or_text = func.replace(cast(plain, Text), " & ", " | ")
    return cast(or_text, _TSQuery())


def _structural_filters(
    *,
    sectors: list[str] | None,
    project_year_min: int | None,
    project_year_max: int | None,
    chunk_types: list[str] | None,
) -> list:
    """Build the shared ``(:filter IS NULL OR …)`` structural predicates.

    Both retrieval branches (dense and lexical) pre-filter on the SAME axes so a
    hybrid query never mixes candidates from different structural scopes.
    """
    sector_col = ChunkRow.metadata_["client_sector"].astext
    year_col = cast(ChunkRow.metadata_["year"].astext, Integer)

    filters = []
    if sectors:
        filters.append(sector_col.in_(sectors))
    if project_year_min is not None:
        filters.append(year_col >= project_year_min)
    if project_year_max is not None:
        filters.append(year_col <= project_year_max)
    if chunk_types:
        filters.append(ChunkRow.chunk_type.in_(chunk_types))
    return filters


class ChunkStore:
    """CRUD + similarity search over ``documents``/``chunks``."""

    async def find_document_id(self, session: AsyncSession, source_path: str) -> int | None:
        """Return the id of the document already ingested from ``source_path``,
        or ``None``. Backs the application-level 409 duplicate guard."""
        stmt = select(DocumentRow.id).where(DocumentRow.source_path == source_path)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def persist_document_with_chunks(
        self,
        session: AsyncSession,
        *,
        source_path: str,
        document_type: str,
        doc_metadata: dict,
        embedded_chunks: list[EmbeddedChunk],
        chunk_type: str = BUDGET_COMPONENT,
    ) -> int:
        """Insert the document row plus all its chunk rows. No commit here —
        the caller's transaction decides when (and whether) anything lands.

        ``chunk_type`` is stamped on every chunk (filterable column); it
        defaults to ``budget_component`` so existing callers are unaffected."""
        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_=doc_metadata,
        )
        session.add(document)
        await session.flush()  # assigns document.id without committing

        session.add_all(
            ChunkRow(
                document_id=document.id,
                chunk_type=chunk_type,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_=chunk.metadata,
            )
            for chunk in embedded_chunks
        )
        return document.id

    async def search(
        self, session: AsyncSession, *, query_vector: list[float], k: int
    ) -> list[Row]:
        """k nearest chunks by cosine distance (``<=>``), sequential scan.

        Cosine over L2/inner product: OpenAI embeddings are normalized so the
        ranking would be equivalent, but cosine keeps us aligned with the RAG
        literature AND with the ``vector_cosine_ops`` operator class of the
        HNSW index the live session adds — operator/index mismatch makes
        Postgres silently ignore the index.
        """
        # distance = ChunkRow.embedding.cosine_distance(query_vector)
        distance = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
            query_vector
        )
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(k)
        )
        return list((await session.execute(stmt)).all())

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> tuple[list[Row], int]:
        """k-NN search with structural pre-filtering and a relevance threshold.

        Session 9 retrieval. Structural filters (sector / project year / chunk
        type) narrow the candidate space BEFORE the vector ranking — the metadata
        is persisted in JSONB (``client_sector``, ``year``) and the ``chunk_type``
        column. Each filter follows the ``(:filter IS NULL OR …)`` pattern: a
        ``None`` filter simply does not apply. The distance threshold then drops
        chunks that are not actually close (no "confidently retrieving garbage").

        Returns
        -------
        tuple[list[Row], int]
            ``(rows, candidates_evaluated)`` where ``rows`` are the top-k chunks
            under the threshold (ascending distance) and ``candidates_evaluated``
            is how many chunks matched the structural filters before the
            threshold/limit were applied.
        """
        structural_filters = _structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )

        distance = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
            query_vector
        )

        count_stmt = select(func.count()).select_from(ChunkRow).where(*structural_filters)
        candidates_evaluated = int((await session.execute(count_stmt)).scalar_one())

        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                distance.label("distance"),
            )
            .where(*structural_filters)
            .where(distance <= distance_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        rows = list((await session.execute(stmt)).all())
        return rows, candidates_evaluated

    async def search_lexical(
        self,
        session: AsyncSession,
        *,
        query_text: str,
        top_k: int = 50,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[Row]:
        """Lexical (keyword) branch of the hybrid retriever — full-text search.

        Ranks chunks by ``ts_rank_cd`` over the generated ``content_tsv`` column
        (``to_tsvector('english', content)``, GIN-indexed). The free-text query
        is turned into an **OR** tsquery (see :func:`_or_tsquery`); ``@@`` keeps
        the chunks matching any salient term before ranking. The ``'english'``
        regconfig MUST match the column's (migration ``0003``) or the GIN index
        is silently ignored.

        Returns the matching rows ordered by descending lexical rank (best first)
        with a ``rank`` column. Same structural pre-filters as the dense branch
        so a hybrid query fuses candidates from one consistent scope. No distance
        threshold here: the dense branch owns the relevance floor; lexical only
        contributes ordering for the fusion.
        """
        tsquery = _or_tsquery(query_text)
        rank = func.ts_rank_cd(ChunkRow.content_tsv, tsquery)

        structural_filters = _structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )

        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                rank.label("rank"),
            )
            .where(*structural_filters)
            .where(ChunkRow.content_tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        return list((await session.execute(stmt)).all())
