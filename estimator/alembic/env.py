"""Alembic migration environment for the Session 6+ persistence layer.

The DB URL is read from the ``DATABASE_URL`` environment variable (with a
fallback to ``app.config.Settings`` for local ``.env`` loading). Migrations are
discovered by importing ``app.foundation.persistence.models``: every SQLAlchemy
model in that module is registered against ``Base.metadata`` and becomes
visible to Alembic's autogenerate.

``pgvector.sqlalchemy.Vector`` is registered on the connection dialect before
running migrations so ``alembic check`` and autogenerate correctly reflect
``vector`` columns instead of producing inconsistent diffs.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

import pgvector.sqlalchemy
from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.foundation.persistence.models import Base  # noqa: F401 — ensure models are imported

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("DATABASE_URL") or get_settings().DATABASE_URL,
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live connection — used by ``alembic upgrade --sql``."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations on a live connection with pgvector type awareness."""
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
