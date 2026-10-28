"""Optional OpenTelemetry traces of the requests the API serves.

The API imports no other project package, so it carries its own small version
of the shared setup (`lab_common.telemetry`): export is off unless
OTEL_EXPORTER_OTLP_ENDPOINT is set and a collector accepts a connection there
at start-up, which with Docker Compose happens only under the `observability`
profile. With export off no provider is installed and the middleware's spans
are no-ops.

`TracingMiddleware` opens one SERVER span per request (health checks
excluded), child of the caller's W3C `traceparent` when there is one. Spans
carry the method, the route template (never the raw path), the status code
and the request id; never headers or bodies.
"""

import os
import socket
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scheduler_api.api.middleware import route_template
from scheduler_api.log import SERVICE

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
PROBE_TIMEOUT_S = 0.5
_HEALTH_PREFIX = "/health/"
_PROPAGATOR = TraceContextTextMapPropagator()

_tracer = trace.get_tracer(__name__)
_provider: TracerProvider | None = None


def collector_endpoint() -> str | None:
    """OTEL_EXPORTER_OTLP_ENDPOINT when a collector listens there, else None."""
    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    url = urlsplit(endpoint)
    if not endpoint or url.hostname is None:
        return None
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        with socket.create_connection((url.hostname, port), timeout=PROBE_TIMEOUT_S):
            pass
    except OSError:
        return None
    return endpoint.rstrip("/")


def setup(span_exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Install the global tracer provider; None (nothing installed) when export is off.

    Once installed, later calls (one per app the factory builds) return it.
    """
    global _provider  # noqa: PLW0603 - one provider per process
    if _provider is not None:
        return _provider
    if span_exporter is None:
        endpoint = collector_endpoint()
        if endpoint is None:
            return None
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415 - only when exporting
            OTLPSpanExporter,
        )

        span_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE}))
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


class TracingMiddleware:
    """One SERVER span per request, named `<method> <route template>`."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].startswith(_HEALTH_PREFIX):
            await self.app(scope, receive, send)
            return

        carrier = {"traceparent": Headers(scope=scope).get("traceparent", "")}
        status: int | None = None

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        with _tracer.start_as_current_span(
            scope["method"],
            context=_PROPAGATOR.extract(carrier),
            kind=SpanKind.SERVER,
            attributes={"http.request.method": scope["method"]},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                await self.app(scope, receive, send_with_status)
            finally:
                route = route_template(scope)
                span.update_name(f"{scope['method']} {route}")
                span.set_attribute("http.route", route)
                request_id = scope.get("state", {}).get("request_id")
                if request_id is not None:
                    span.set_attribute("lab.request_id", request_id)
                if status is not None:
                    span.set_attribute("http.response.status_code", status)
                if status is None or status >= 500:  # noqa: PLR2004 - server errors
                    span.set_status(Status(StatusCode.ERROR))
