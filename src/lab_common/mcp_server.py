"""Pieces shared by the MCP servers (OCR and RAG): bearer auth, log scrubbing and
the request id the logs are correlated by.

`BearerAuthMiddleware` wraps a server's whole ASGI app: every path except the
public ones (`/health`) requires `Authorization: Bearer <token>`, so both
`/sse` and `/messages/` are covered. The token is compared in constant time;
the 401 body and the log line never include what the client sent.

`LibraryLogFilter` scrubs records from the `mcp` and `uvicorn` libraries:
their error messages can quote the request that caused them. A malformed
JSON-RPC call, for instance, is logged by `mcp` as a pydantic error whose
`input_value` holds part of the body (an image, exam mentions), and
`mcp.shared.session` logs the whole rejected message through the root logger.
Those records keep their event text up to the first ": " and the exception
type only. A record counts as coming from a library when its logger is named
after it or when it was emitted from a file inside the library (which catches
calls on the root logger).

Plain ASGI and stdlib only, so `lab_common` does not depend on the server
libraries; the libraries are located without importing them.
"""

import hmac
import importlib.util
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Any

from lab_common import logging as lab_logging

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REDACTED = "[REDACTED:PAYLOAD]"
LIBRARY_LOGGERS = ("mcp", "uvicorn")
MCP_PATHS = frozenset({"/sse", "/messages/"})
_UNAUTHORIZED_BODY = json.dumps(
    {"code": "UNAUTHORIZED", "message": "missing or invalid bearer token"},
    separators=(",", ":"),
).encode()

log = lab_logging.get_logger(__name__)


def logged_request_id(value: object) -> str | None:
    """`value` if it is a UUID (the agent sends its run id), else None: it is client input."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        return None
    return str(value) if isinstance(value, str) else None


class BearerAuthMiddleware:
    """Rejects HTTP requests without the bearer token, except on public paths."""

    def __init__(self, app: ASGIApp, token: str, public_paths: frozenset[str]) -> None:
        self._app = app
        self._expected = f"bearer {token}".encode()
        self._public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._public_paths:
            await self._app(scope, receive, send)
            return
        if not self._authorized(scope):
            # Unknown paths are logged as null: a path is client input.
            log.warning("auth_failed", path=scope["path"] if scope["path"] in MCP_PATHS else None)
            await _unauthorized(send)
            return
        await self._app(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        values = [v for k, v in scope["headers"] if k == b"authorization"]
        if len(values) != 1:
            return False
        scheme, _, credentials = values[0].partition(b" ")
        # The scheme is case-insensitive (RFC 9110); the token is not.
        return hmac.compare_digest(scheme.lower() + b" " + credentials, self._expected)


async def _unauthorized(send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def _library_dirs(names: Sequence[str]) -> tuple[str, ...]:
    dirs = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            dirs.append(os.path.dirname(os.path.abspath(spec.origin)) + os.sep)
    return tuple(dirs)


class LibraryLogFilter(logging.Filter):
    """Drops request payloads that `mcp` and `uvicorn` put into their log records."""

    def __init__(self, libraries: Sequence[str] = LIBRARY_LOGGERS) -> None:
        super().__init__()
        self._libraries = tuple(libraries)
        self._dirs = _library_dirs(self._libraries)

    def _from_library(self, record: logging.LogRecord) -> bool:
        name = record.name
        if any(name == lib or name.startswith(f"{lib}.") for lib in self._libraries):
            return True
        return bool(self._dirs) and os.path.abspath(record.pathname).startswith(self._dirs)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._from_library(record):
            return True
        # uvicorn duplicates its message with ANSI colors for its own formatter.
        record.__dict__.pop("color_message", None)
        # Set by any handler that formatted the record before ours; structlog
        # would pick it up as an extra field holding the unredacted message.
        record.__dict__.pop("message", None)
        if record.exc_info:
            exc_type = record.exc_info[0]
            record.exception_type = exc_type.__name__ if exc_type else None
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            record.exception = REDACTED
        # Messages are often f-strings that embed an exception or the request
        # after a colon ("Received exception from stream: <pydantic error with
        # the body>"), at any level, including debug dumps of the message.
        message = record.getMessage()
        head, separator, _ = message.partition(": ")
        record.msg = f"{head}{separator}{REDACTED}" if separator else message
        record.args = None
        return True
