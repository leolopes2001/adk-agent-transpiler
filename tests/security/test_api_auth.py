import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scheduler_api import log
from scheduler_api.api import deps
from scheduler_api.main import create_app
from scheduler_api.settings import Settings

pytestmark = pytest.mark.security

CATALOG = Path(__file__).parents[2] / "data" / "exams_catalog.json"
API_KEY = "security-test-key-0123456789abcdefghijk"
PROTECTED = "/_test/protected"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    if log._handler is not None:
        root.removeHandler(log._handler)
        log._handler = None
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture
def app() -> FastAPI:
    app = create_app(
        Settings(
            database_url="postgresql+psycopg://sec:sec@127.0.0.1:1/sec",
            scheduler_api_key=API_KEY,
            catalog_path=CATALOG,
        )
    )

    # Stand-in for the /v1 routes: same dependency, no database involved.
    @app.get(PROTECTED, dependencies=[Depends(deps.require_api_key)])
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def test_accepts_the_configured_key(client: TestClient) -> None:
    response = client.get(PROTECTED, headers={"X-API-Key": API_KEY})

    assert response.status_code == 200


def test_rejects_missing_key(client: TestClient) -> None:
    response = client.get(PROTECTED)

    assert response.status_code == 401


@pytest.mark.parametrize(
    "key",
    [
        "wrong-key-0123456789abcdefghijklmnopqrstu",
        API_KEY[:-1],
        API_KEY + "x",
        API_KEY.upper(),
        f" {API_KEY}",
        "",
    ],
    ids=["wrong", "prefix", "suffix", "case", "padded", "empty"],
)
def test_rejects_wrong_key(client: TestClient, key: str) -> None:
    response = client.get(PROTECTED, headers={"X-API-Key": key})

    assert response.status_code == 401


def test_rejects_non_ascii_key_without_crashing(client: TestClient) -> None:
    response = client.get(PROTECTED, headers={"X-API-Key": "chave-inválida".encode()})

    assert response.status_code == 401


def test_key_is_compared_in_constant_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = deps.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(deps.hmac, "compare_digest", spy)
    client.get(PROTECTED, headers={"X-API-Key": "wrong"})

    assert calls == [(b"wrong", API_KEY.encode())]


@pytest.mark.parametrize("path", ["/health/live", "/health/ready", "/docs", "/openapi.json"])
def test_public_endpoints_need_no_key(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code != 401


def test_openapi_declares_the_api_key_scheme(client: TestClient) -> None:
    schemes = client.get("/openapi.json").json()["components"]["securitySchemes"]

    assert schemes["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Shared API key configured on the server (`SCHEDULER_API_KEY`).",
    }


def test_keys_are_never_logged(app: FastAPI, client: TestClient) -> None:
    stream = io.StringIO()
    log.configure_logging(stream=stream)
    wrong = "attempted-key-0123456789abcdefghijklmno"

    client.get(PROTECTED, headers={"X-API-Key": wrong})
    client.get(PROTECTED, headers={"X-API-Key": API_KEY})

    output = stream.getvalue()
    assert '"auth_failed"' in output
    assert wrong not in output
    assert API_KEY not in output
