import json
import logging
import sys
from typing import Any

import pytest

from lab_common.mcp_server import (
    REDACTED,
    BearerAuthMiddleware,
    LibraryLogFilter,
    Message,
    Receive,
    Scope,
    Send,
)

pytestmark = pytest.mark.unit

TOKEN = "t" * 32


async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def receive() -> Message:
    return {"type": "http.request", "body": b""}


async def run(path: str, headers: list[tuple[bytes, bytes]], scope_type: str = "http") -> Any:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    app = BearerAuthMiddleware(downstream, TOKEN, frozenset({"/health"}))
    await app({"type": scope_type, "path": path, "headers": headers}, receive, send)
    return sent


def bearer(token: str = TOKEN, scheme: bytes = b"Bearer") -> list[tuple[bytes, bytes]]:
    return [(b"authorization", scheme + b" " + token.encode())]


class TestBearerAuthMiddleware:
    async def test_valid_token_passes(self) -> None:
        sent = await run("/sse", bearer())
        assert sent[0]["status"] == 200

    async def test_scheme_is_case_insensitive(self) -> None:
        sent = await run("/sse", bearer(scheme=b"BEARER"))
        assert sent[0]["status"] == 200

    @pytest.mark.parametrize(
        "headers",
        [[], bearer("x" * 32), bearer(scheme=b"Basic"), bearer() + bearer()],
    )
    async def test_rejects_with_a_json_401(self, headers: list[tuple[bytes, bytes]]) -> None:
        start, body = await run("/messages/", headers)

        assert start["status"] == 401
        response_headers = dict(start["headers"])
        assert response_headers[b"www-authenticate"] == b"Bearer"
        assert response_headers[b"content-type"] == b"application/json"
        assert int(response_headers[b"content-length"]) == len(body["body"])
        assert json.loads(body["body"]) == {
            "code": "UNAUTHORIZED",
            "message": "missing or invalid bearer token",
        }

    async def test_public_path_needs_no_token(self) -> None:
        sent = await run("/health", [])
        assert sent[0]["status"] == 200

    async def test_non_http_scopes_pass_through(self) -> None:
        sent = await run("/", [], scope_type="lifespan")
        assert sent[0]["status"] == 200


def record(name: str, message: str, pathname: str = __file__) -> logging.LogRecord:
    return logging.LogRecord(name, logging.ERROR, pathname, 1, message, None, None)


class TestLibraryLogFilter:
    def test_custom_library_list(self) -> None:
        rec = record("httpx", "Request failed: body")

        LibraryLogFilter(["httpx"]).filter(rec)

        assert rec.getMessage() == f"Request failed: {REDACTED}"

    def test_missing_library_is_matched_by_name_only(self) -> None:
        log_filter = LibraryLogFilter(["not_an_installed_package"])
        rec = record("not_an_installed_package.sub", "boom: secret")
        other = record("app", "a: b")

        log_filter.filter(rec)
        log_filter.filter(other)

        assert rec.getMessage() == f"boom: {REDACTED}"
        assert other.getMessage() == "a: b"

    def test_exception_keeps_only_its_type(self) -> None:
        try:
            raise ValueError("input_value='secret'")
        except ValueError:
            rec = logging.LogRecord(
                "mcp.server", logging.ERROR, __file__, 1, "failed", None, exc_info=sys.exc_info()
            )

        LibraryLogFilter().filter(rec)

        assert rec.exc_info is None
        assert rec.__dict__["exception_type"] == "ValueError"
        assert rec.__dict__["exception"] == REDACTED
