"""Vector store — PostgreSQL + pgvector persistence for embedded chunks."""

from app.generation.rag.store.models import Base, ChunkRow, DocumentRow
from app.generation.rag.store.search import search_similar_chunks

__all__ = ["Base", "ChunkRow", "DocumentRow", "search_similar_chunks"]
