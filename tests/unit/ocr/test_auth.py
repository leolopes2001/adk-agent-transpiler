import io
import json
import logging
from collections.abc import Iterator

import pytest
import structlog
from ocr_fakes import TOKEN, FakeOcrEngine, make_settings
from starlette.testclient import TestClient

from lab_common import logging as lab_logging
from ocr_mcp.server import build_app

pytestmark = pytest.mark.unit

MCP_PATHS = ["/sse", "/messages/"]


@pytest.fixture(autouse=True)
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    lab_logging.configure("ocr-mcp", stream=stream)
    yield stream
    if lab_logging._handler is not None:
        logging.getLogger().removeHandler(lab_logging._handler)
        lab_logging._handler = None
    structlog.reset_defaults()


class BrokenVersion(FakeOcrEngine):
    def version(self) -> str:
        raise RuntimeError("tesseract missing")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app(make_settings(), FakeOcrEngine())) as test_client:
        yield test_client


def auth(token: str = TOKEN, scheme: str = "Bearer") -> dict[str, str]:
    return {"Authorization": f"{scheme} {token}"}


class TestBearerToken:
    @pytest.mark.parametrize("path", MCP_PATHS)
    def test_missing_token_is_rejected(self, client: TestClient, path: str) -> None:
        response = client.request("GET" if path == "/sse" else "POST", path)

        assert response.status_code == 401
        assert response.json() == {
            "code": "UNAUTHORIZED",
            "message": "missing or invalid bearer token",
        }
        assert response.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize(
        "headers",
        [
            auth("wrong-token-" + "x" * 32),
            auth(TOKEN[:-1]),
            auth(TOKEN + "x"),
            auth(TOKEN, scheme="Basic"),
            auth(TOKEN.upper()),
            {"Authorization": TOKEN},
            {"Authorization": f"Bearer  {TOKEN}"},
        ],
    )
    @pytest.mark.parametrize("path", MCP_PATHS)
    def test_wrong_token_is_rejected(
        self, client: TestClient, path: str, headers: dict[str, str]
    ) -> None:
        response = client.request("GET" if path == "/sse" else "POST", path, headers=headers)
        assert response.status_code == 401

    def test_duplicate_authorization_headers_are_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/messages/",
            headers=[("Authorization", f"Bearer {TOKEN}"), ("Authorization", "Bearer x")],
        )
        assert response.status_code == 401

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    def test_valid_token_reaches_the_mcp_app(self, client: TestClient, scheme: str) -> None:
        # Without a session id the SSE transport answers 400: the request got past auth.
        response = client.post("/messages/", headers=auth(scheme=scheme))
        assert response.status_code == 400

    def test_unknown_paths_also_require_the_token(self, client: TestClient) -> None:
        assert client.get("/anything").status_code == 401
        assert client.get("/anything", headers=auth()).status_code == 404

    def test_rejection_is_logged_without_credentials(
        self, client: TestClient, log_stream: io.StringIO
    ) -> None:
        client.post("/messages/?session_id=abc", headers=auth("leaked-" + "z" * 32))

        logs = log_stream.getvalue()
        assert "leaked" not in logs
        # Other records here come from the test's HTTP client, not from the server.
        [record] = [json.loads(line) for line in logs.splitlines() if "auth_failed" in line]
        assert record["path"] == "/messages/"
        assert set(record) == {"event", "path", "level", "ts", "service"}

    def test_response_does_not_echo_the_token(self, client: TestClient) -> None:
        response = client.get("/sse", headers=auth("echo-me-" + "z" * 32))
        assert "echo-me" not in response.text


class TestHealth:
    def test_is_public(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "engine": "fake",
            "lang": "por",
            "version": "0.0-test",
        }

    def test_works_with_a_token_too(self, client: TestClient) -> None:
        assert client.get("/health", headers=auth()).status_code == 200

    def test_only_get(self, client: TestClient) -> None:
        assert client.post("/health").status_code == 405

    def test_engine_failure_is_unavailable(self, log_stream: io.StringIO) -> None:
        with TestClient(build_app(make_settings(), BrokenVersion())) as client:
            response = client.get("/health")

        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
        assert '"error_type": "RuntimeError"' in log_stream.getvalue()
