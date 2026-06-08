"""Vector store — PostgreSQL + pgvector persistence for embedded chunks."""

from app.generation.rag.store.models import Base, ChunkRow, DocumentRow

__all__ = ["Base", "ChunkRow", "DocumentRow"]
