"""Session 10 — full-text search: generated ``content_tsv`` column + GIN index.

Revision ID: 0003_session10_fts
Revises: 0002_session8_pgvector
Create Date: 2026-06-23 00:00:00

The lexical branch of the hybrid retriever needs full-text search over the
chunk content. Two pieces, both created here:

* ``content_tsv`` — a **generated, STORED** column: Postgres recomputes
  ``to_tsvector('english', content)`` on every insert/update of ``content``
  and persists it. No application code, no trigger, no chance of the tsvector
  drifting out of sync with the text it indexes.
* ``ix_chunks_content_tsv`` — a **GIN** index over that column, the structure
  that makes ``@@`` lookups and ``ts_rank_cd`` ranking fast.

The ``'english'`` regconfig is a deliberate, load-bearing choice: the sample
corpus is in English, and the SAME config MUST be used by the query side
(``plainto_tsquery('english', …)``). A mismatch between the column's config and
the query's config makes Postgres silently ignore the GIN index — correct
results, but a sequential scan. If the corpus ever switches to Spanish, change
the regconfig here AND in the repository's query, then re-run the migration.

The expression is immutable (the regconfig is a literal, not a column), which
is what lets Postgres accept it as a ``GENERATED ALWAYS AS … STORED`` column.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003_session10_fts"
down_revision: Union[str, None] = "0002_session8_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
