"""Request correlation and access logging.

Pure ASGI middleware instead of `BaseHTTPMiddleware`, so the bound context
variables reach the endpoint (and the thread pool that runs sync endpoints).
"""

import re
import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scheduler_api.log import get_logger

REQUEST_ID_HEADER = "X-Request-ID"

# Anything else is replaced by a fresh id: the value is echoed in a header and
# written to every log line, so it must not carry arbitrary client text.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9_\-:.]{1,128}")

_log = get_logger(__name__)


def _request_id(scope: Scope) -> str:
    incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
    if incoming is not None and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return str(uuid.uuid4())


def _route_template(scope: Scope) -> str:
    # Route templates, not raw paths, so unmatched URLs never end up in the logs.
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "<unmatched>"


class RequestContextMiddleware:
    """Assigns a request id, echoes it in the response and logs one `request` event."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_request_id)
            finally:
                _log.info(
                    "request",
                    method=scope["method"],
                    route=_route_template(scope),
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
