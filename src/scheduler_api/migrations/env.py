"""Alembic environment.

The URL comes from, in order: a connection passed in `config.attributes`
(tests), the `sqlalchemy.url` option, or the `DATABASE_URL` environment
variable (container entrypoint and local use).

Logging uses the API's JSON setup, so the container log has a single format
from the migrations on; tests turn it off with `configure_logger = False`.
"""

import os

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from scheduler_api.infra.orm import Base
from scheduler_api.log import configure_logging

config = context.config
target_metadata = Base.metadata

if config.attributes.get("configure_logger", True):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("set DATABASE_URL to run the migrations")
    return url


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        _run(connection)
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
