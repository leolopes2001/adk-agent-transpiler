import hmac
import importlib.util
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import Any

from lab_common import logging as lab_logging

REDACTED = "[REDACTED:PAYLOAD]"
LIBRARY_LOGGERS = ("mcp", "uvicorn")
MCP_PATHS = frozenset({"/sse", "/messages/"})
_UNAUTHORIZED_BODY = json.dumps(
    {"code": "UNAUTHORIZED", "message": "missing or invalid bearer token"},
    separators=(",", ":"),
).encode()

# Plain ASGI types, so lab_common does not depend on the server libraries.
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

log = lab_logging.get_logger(__name__)


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, token: str, public_paths: frozenset[str]) -> None:
        self._app = app
        self._expected_credentials = f"bearer {token}".encode()
        self._public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._public_paths:
            await self._app(scope, receive, send)
            return

        if not self._is_authorized(scope):
            # Unknown paths are logged as null: a path is client input.
            log.warning("auth_failed", path=scope["path"] if scope["path"] in MCP_PATHS else None)
            await _send_unauthorized(send)
            return

        await self._app(scope, receive, send)

    def _is_authorized(self, scope: Scope) -> bool:
        values = [v for k, v in scope["headers"] if k == b"authorization"]
        if len(values) != 1:
            return False

        scheme, _, credentials = values[0].partition(b" ")
        # The scheme is case-insensitive (RFC 9110); the token is not.
        return hmac.compare_digest(scheme.lower() + b" " + credentials, self._expected_credentials)


class LibraryLogFilter(logging.Filter):
    def __init__(self, libraries: Sequence[str] = LIBRARY_LOGGERS) -> None:
        super().__init__()
        self._libraries = tuple(libraries)
        self._source_dirs = _library_dirs(self._libraries)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._is_from_library(record):
            return True

        # uvicorn duplicates its message with ANSI colors for its own formatter.
        record.__dict__.pop("color_message", None)
        # Left by a handler that formatted the record first; structlog would log it unredacted.
        record.__dict__.pop("message", None)
        _drop_exception(record)
        _redact_message(record)
        return True

    def _is_from_library(self, record: logging.LogRecord) -> bool:
        name = record.name
        if any(name == lib or name.startswith(f"{lib}.") for lib in self._libraries):
            return True

        if not self._source_dirs:
            return False

        # The file check catches libraries that log through the root logger (mcp.shared.session).
        return os.path.abspath(record.pathname).startswith(self._source_dirs)


def logged_request_id(value: object) -> str | None:
    # Only a UUID (the agent's run id) is logged: the request id is client input.
    try:
        uuid.UUID(str(value))
    except ValueError:
        return None

    return str(value) if isinstance(value, str) else None


async def _send_unauthorized(send: Send) -> None:
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


def _drop_exception(record: logging.LogRecord) -> None:
    if not record.exc_info:
        return

    exc_type = record.exc_info[0]
    record.exception_type = exc_type.__name__ if exc_type else None
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None
    record.exception = REDACTED


def _redact_message(record: logging.LogRecord) -> None:
    # Messages often embed the request or an exception after ": ", at any level.
    message = record.getMessage()
    head, separator, _ = message.partition(": ")
    record.msg = f"{head}{separator}{REDACTED}" if separator else message
    record.args = None
