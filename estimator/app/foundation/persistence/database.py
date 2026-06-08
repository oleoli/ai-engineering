"""SQLAlchemy engine, session factory and per-request session helpers.

The sync API backs Session 6 ingestion BackgroundTasks. The async API backs
Session 8 embedding persistence (``POST /embeddings/ingest``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _async_database_url(sync_url: str) -> str:
    """Derive an asyncpg URL from the sync psycopg URL in settings."""
    if "+psycopg" in sync_url:
        return sync_url.replace("+psycopg", "+asyncpg", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return sync_url


@lru_cache
def create_engine_from_settings() -> Engine:
    """Build the global engine from ``Settings.DATABASE_URL`` (singleton)."""
    return create_engine(
        get_settings().DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Build the global async engine (singleton)."""
    return create_async_engine(
        _async_database_url(get_settings().DATABASE_URL),
        pool_pre_ping=True,
    )


SessionLocal = sessionmaker(
    bind=create_engine_from_settings(),
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=create_async_engine_from_settings(),
    autoflush=False,
    expire_on_commit=False,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a Session and closes it on exit."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession within a context manager."""
    async with AsyncSessionLocal() as session:
        yield session
