import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from scheduler_api.infra.orm import Base
from scheduler_api.main import create_app
from scheduler_api.settings import Settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[3]
TABLES = {"appointments", "appointment_exams", "idempotency_keys"}


@pytest.fixture
def engine(empty_database: str) -> Iterator[Engine]:
    engine = create_engine(empty_database)
    yield engine
    engine.dispose()


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_schema_from_scratch(
    empty_database: str, engine: Engine, alembic_config: Callable[[str], Config]
) -> None:
    command.upgrade(alembic_config(empty_database), "head")

    assert _tables(engine) == TABLES
    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    assert version == "0001_initial"


def test_orm_matches_migrated_schema(
    empty_database: str, engine: Engine, alembic_config: Callable[[str], Config]
) -> None:
    command.upgrade(alembic_config(empty_database), "head")

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"compare_type": True, "compare_server_default": True}
        )
        assert compare_metadata(context, Base.metadata) == []


def test_downgrade_then_upgrade_again(
    empty_database: str, engine: Engine, alembic_config: Callable[[str], Config]
) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    assert _tables(engine) == set()

    command.upgrade(config, "head")
    assert _tables(engine) == TABLES


def test_status_is_restricted_to_requested(
    empty_database: str, engine: Engine, alembic_config: Callable[[str], Config]
) -> None:
    command.upgrade(alembic_config(empty_database), "head")
    insert = text(
        "INSERT INTO appointments (id, status, patient_ref, source_document_sha256, requested_by)"
        " VALUES ('apt_1', :status, 'pt_x', :sha, 'tester')"
    )

    with engine.begin() as connection:
        connection.execute(insert, {"status": "requested", "sha": "a" * 64})
    with pytest.raises(IntegrityError, match="ck_appointments_status"), engine.begin() as conn:
        conn.execute(insert, {"status": "scheduled", "sha": "a" * 64})


def test_alembic_cli_reads_database_url_from_environment(
    empty_database: str, engine: Engine
) -> None:
    # Same command the container entrypoint runs.
    env = {**os.environ, "DATABASE_URL": empty_database}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert _tables(engine) == TABLES


def test_ready_checks_the_real_database(
    empty_database: str, settings_for: Callable[[str], Settings]
) -> None:
    with TestClient(create_app(settings_for(empty_database))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["database"] == "ok"


def test_ready_reports_unreachable_database(
    empty_database: str, settings_for: Callable[[str], Settings]
) -> None:
    unreachable = empty_database.rsplit("/", 1)[0] + "/does_not_exist"
    with TestClient(create_app(settings_for(unreachable))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["database"] == "error"
