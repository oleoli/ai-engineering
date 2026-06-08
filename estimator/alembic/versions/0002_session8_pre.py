from __future__ import annotations

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision: str = "0002_session8_pre"
down_revision: Union[str, None] = "0001_session6_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", postgresql.JSONB,
                  server_default="{}", nullable=False),
    )
    op.create_index("ix_documents_source_path", "documents", ["source_path"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("document_id", sa.BigInteger,
                  sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("chunk_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("metadata", postgresql.JSONB,
                  server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index("ix_chunks_metadata_gin", "chunks", ["metadata"],
                    postgresql_using="gin")