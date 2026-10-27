"""Postgres for the API integration tests.

One container per session (testcontainers, removed at the end); each test that
needs a schema gets its own empty database, so tests never see each other's rows.
"""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import URL, Engine, create_engine, make_url, text
from testcontainers.community.postgres import PostgresContainer

from scheduler_api.settings import Settings

ROOT = Path(__file__).parents[3]
ALEMBIC_INI = ROOT / "alembic.ini"
POSTGRES_IMAGE = "postgres:16-alpine"
CATALOG = ROOT / "data" / "exams_catalog.json"
API_KEY = "integration-test-key-0123456789abcdefghij"


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    with PostgresContainer(POSTGRES_IMAGE, driver="psycopg") as container:
        yield container


@pytest.fixture(scope="session")
def admin_engine(postgres: PostgresContainer) -> Iterator[Engine]:
    engine = create_engine(postgres.get_connection_url(), isolation_level="AUTOCOMMIT")
    yield engine
    engine.dispose()


@pytest.fixture
def empty_database(postgres: PostgresContainer, admin_engine: Engine) -> Iterator[str]:
    """URL of a freshly created database, dropped after the test."""
    name = f"t_{uuid.uuid4().hex}"
    with admin_engine.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url: URL = make_url(postgres.get_connection_url()).set(database=name)
    yield url.render_as_string(hide_password=False)
    with admin_engine.connect() as connection:
        connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture
def alembic_config() -> Callable[[str], Config]:
    def build(url: str) -> Config:
        config = Config(ALEMBIC_INI)
        config.set_main_option("sqlalchemy.url", url)
        # fileConfig would replace the logging setup the tests rely on.
        config.attributes["configure_logger"] = False
        return config

    return build


@pytest.fixture
def settings_for() -> Callable[[str], Settings]:
    def build(database_url: str) -> Settings:
        return Settings(
            database_url=database_url,
            scheduler_api_key=API_KEY,
            catalog_path=CATALOG,
        )

    return build
