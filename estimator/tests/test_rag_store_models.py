"""Schema-level tests for the Session 8 ORM models.

Pure metadata introspection — no engine, no database. These pin the design
decisions the exercise asks students to defend: the reserved-name mapping
(``metadata_`` → column ``"metadata"``), the 1536-dim nullable vector, the
CASCADE FK and the deliberate absence of a vector index.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector

from app.generation.rag.store.models import ChunkRow, DocumentRow


def test_metadata_attribute_maps_to_metadata_column():
    assert DocumentRow.metadata_.expression.name == "metadata"
    assert ChunkRow.metadata_.expression.name == "metadata"


def test_embedding_is_a_nullable_1536_dim_vector():
    embedding = ChunkRow.__table__.c.embedding
    assert isinstance(embedding.type, Vector)
    assert embedding.type.dim == 1536
    # Nullable on purpose: leaves the door open to insert-now-embed-later.
    assert embedding.nullable is True


def test_chunks_fk_cascades_on_document_delete():
    fk = next(iter(ChunkRow.__table__.c.document_id.foreign_keys))
    assert fk.column.table.name == "documents"
    assert fk.ondelete == "CASCADE"


def test_no_vector_index_yet():
    # The live session adds HNSW; the previo ships sequential scan only.
    index_columns = {col.name for index in ChunkRow.__table__.indexes for col in index.columns}
    assert "embedding" not in index_columns


def test_relational_indexes_present():
    index_names = {index.name for index in ChunkRow.__table__.indexes}
    assert index_names == {
        "ix_chunks_document_id",
        "ix_chunks_chunk_type",
        "ix_chunks_metadata_gin",
        # Session 10: GIN index over the generated content_tsv column (FTS).
        "ix_chunks_content_tsv",
    }
    assert {index.name for index in DocumentRow.__table__.indexes} == {"ix_documents_source_path"}
