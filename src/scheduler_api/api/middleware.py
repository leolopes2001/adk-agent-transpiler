import re
import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scheduler_api.api.errors import internal_error_response
from scheduler_api.log import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
HEALTH_PREFIX = "/health/"

# Echoed in a header and in every log line, so it must not carry arbitrary client text.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9_\-:.]{1,128}")

_log = get_logger(__name__)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        status: int | None = None
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
            except Exception:
                _log.exception("unhandled_error", route=route_template(scope))
                # Headers already sent: nothing sensible to answer; let the server close it.
                if status is not None:
                    raise

                await internal_error_response(scope)(scope, receive, send_with_request_id)
            finally:
                _log_request(scope, status, started)


def route_template(scope: Scope) -> str:
    route_path = getattr(scope.get("route"), "path", None)
    if isinstance(route_path, str):
        return route_path

    # Only a matched route's path is logged: an unmatched URL is arbitrary client text.
    if scope.get("endpoint") is not None:
        return str(scope["path"])

    return "<unmatched>"


def _resolve_request_id(scope: Scope) -> str:
    incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
    if incoming is not None and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming

    return str(uuid.uuid4())


def _log_request(scope: Scope, status: int | None, started: float) -> None:
    route = route_template(scope)
    # Health checks run every few seconds and would drown the log.
    if route.startswith(HEALTH_PREFIX):
        return

    _log.info(
        "request",
        method=scope["method"],
        route=route,
        status=status,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
