import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scheduler_api.log import get_logger
from scheduler_api.main import create_app
from scheduler_api.settings import Settings

pytestmark = pytest.mark.unit

type LogRecords = Callable[[], list[dict[str, Any]]]


def test_live_is_ok(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {}}


def test_ready_reports_each_check(app: FastAPI) -> None:
    with TestClient(app) as client:
        app.state.readiness_checks = {"database": lambda: True, "catalog": lambda: True}
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok", "catalog": "ok"}}


def test_ready_is_unavailable_when_a_check_fails(app: FastAPI) -> None:
    def broken() -> bool:
        raise ConnectionError("db down")

    with TestClient(app) as client:
        app.state.readiness_checks = {"database": broken, "catalog": lambda: True}
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "checks": {"database": "error", "catalog": "ok"},
    }
    assert "db down" not in response.text


def test_ready_is_unavailable_when_a_check_returns_false(app: FastAPI) -> None:
    with TestClient(app) as client:
        app.state.readiness_checks = {"catalog": lambda: False}
        response = client.get("/health/ready")

    assert response.status_code == 503


def test_generates_request_id_when_absent(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert uuid.UUID(response.headers["X-Request-ID"]).version == 4


def test_echoes_valid_request_id(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "run-42:abc.DEF_9"})

    assert response.headers["X-Request-ID"] == "run-42:abc.DEF_9"


@pytest.mark.parametrize("value", ["has space", "x" * 129, 'quote"d', "novação", ""])
def test_replaces_invalid_request_id(app: FastAPI, value: str) -> None:
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": value.encode()})

    echoed = response.headers["X-Request-ID"]
    assert echoed != value
    assert uuid.UUID(echoed).version == 4


def test_logs_one_request_event_with_route_template(app: FastAPI, log_records: LogRecords) -> None:
    with TestClient(app) as client:
        client.get("/health/live", headers={"X-Request-ID": "req-1234"})

    [event] = [r for r in log_records() if r["event"] == "request"]
    assert event["service"] == "scheduler-api"
    assert event["request_id"] == "req-1234"
    assert event["method"] == "GET"
    assert event["route"] == "/health/live"
    assert event["status"] == 200
    assert isinstance(event["duration_ms"], float)
    assert {"ts", "level"} <= event.keys()


def test_unmatched_path_is_not_logged(app: FastAPI, log_records: LogRecords) -> None:
    with TestClient(app) as client:
        response = client.get("/not-a-route/secret-looking-value")

    assert response.status_code == 404
    # The test client's own httpx logger records the URL; only server records matter.
    server = [r for r in log_records() if not r["event"].startswith("HTTP Request:")]
    [event] = [r for r in server if r["event"] == "request"]
    assert event["route"] == "<unmatched>"
    assert "secret-looking-value" not in json.dumps(server)


def test_forbidden_keys_are_dropped_from_logs(log_records: LogRecords) -> None:
    get_logger("test").info("oops", body='{"patient_ref": "x"}', kept="yes")

    [event] = log_records()
    assert "body" not in event
    assert event["kept"] == "yes"


def test_settings_require_a_long_api_key(settings: Settings) -> None:
    with pytest.raises(ValidationError, match="scheduler_api_key"):
        Settings(
            database_url=settings.database_url,
            scheduler_api_key="short",
            catalog_path=settings.catalog_path,
        )


def test_settings_hide_secrets(settings: Settings) -> None:
    text = repr(settings) + str(settings)

    assert settings.scheduler_api_key.get_secret_value() not in text
    assert "unit:unit" not in text


def test_create_app_reads_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://env:env@127.0.0.1:1/env")
    monkeypatch.setenv("SCHEDULER_API_KEY", "e" * 32)
    monkeypatch.setenv("CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("IDEMPOTENCY_TTL_HOURS", "6")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    settings: Settings = create_app().state.settings

    assert settings.scheduler_api_key.get_secret_value() == "e" * 32
    assert settings.catalog_path == tmp_path / "catalog.json"
    assert settings.idempotency_ttl_hours == 6
    assert settings.log_level == "WARNING"
