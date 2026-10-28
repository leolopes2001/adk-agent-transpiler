"""Optional OpenTelemetry traces of the requests the API serves, and its logs.

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

`OtlpLogExport` sends each rendered log line (see `scheduler_api.log`) as an
OTLP log record whose body is that exact line, with its top-level scalar
fields as attributes and the trace and span id of the request. Its logger
provider is private to this export; the SDK's own records are not exported.
"""

import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import AttributeValue
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scheduler_api.api.middleware import route_template
from scheduler_api.log import SERVICE

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
PROBE_TIMEOUT_S = 0.5
STARTUP_WAIT_S = 10.0
"""How long to wait for a collector whose name resolves (it may still be starting)."""
_HEALTH_PREFIX = "/health/"
_PROPAGATOR = TraceContextTextMapPropagator()

_tracer = trace.get_tracer(__name__)
_SEVERITY = {
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}


class OtlpLogExport:
    """`scheduler_api.log.LogExport` that sends each rendered line as an OTLP log record."""

    def __init__(self, provider: LoggerProvider) -> None:
        self._logger = provider.get_logger("scheduler_api.log")

    def __call__(self, record: logging.LogRecord, line: str) -> None:
        # The SDK logs its own export failures; exporting those would feed the failure.
        if record.name.startswith("opentelemetry"):
            return
        self._logger.emit(
            LogRecord(
                timestamp=int(record.created * 1e9),
                context=context_api.get_current(),
                severity_text=record.levelname,
                severity_number=_SEVERITY.get(record.levelname, SeverityNumber.UNSPECIFIED),
                body=line,
                attributes=_scalar_fields(line),
            )
        )


def _scalar_fields(line: str) -> dict[str, AttributeValue]:
    try:
        fields = json.loads(line)
    except ValueError:
        return {}
    if not isinstance(fields, dict):
        return {}
    return {
        key: value
        for key, value in fields.items()
        if key != "ts" and isinstance(value, str | int | float | bool)
    }


@dataclass(frozen=True)
class Telemetry:
    tracer_provider: TracerProvider
    log_export: OtlpLogExport


_installed: Telemetry | None = None


def collector_endpoint(wait_s: float = STARTUP_WAIT_S) -> str | None:
    """OTEL_EXPORTER_OTLP_ENDPOINT when a collector listens there, else None.

    A name that does not resolve (no `observability` profile) is off at once. A
    name that resolves belongs to a collector that may still be starting, so the
    port is tried for up to `wait_s` seconds.
    """
    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    url = urlsplit(endpoint)
    if not endpoint or url.hostname is None:
        return None
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        socket.getaddrinfo(url.hostname, port)
    except OSError:
        return None
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with socket.create_connection((url.hostname, port), timeout=PROBE_TIMEOUT_S):
                return endpoint.rstrip("/")
        except OSError:
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)


def setup(
    span_exporter: SpanExporter | None = None, log_exporter: LogRecordExporter | None = None
) -> Telemetry | None:
    """Install the global tracer provider and a log export; None when export is off.

    Once installed, later calls (one per app the factory builds) return it.
    Without exporters, spans and logs go to the collector, if one is there.
    """
    global _installed  # noqa: PLW0603 - one provider per process
    if _installed is not None:
        return _installed
    if span_exporter is None or log_exporter is None:
        endpoint = collector_endpoint()
        if endpoint is None:
            return None
        # Only when exporting: the exporters pull in protobuf and requests.
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (  # noqa: PLC0415
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        span_exporter = span_exporter or OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        log_exporter = log_exporter or OTLPLogExporter(endpoint=f"{endpoint}/v1/logs")
    resource = Resource.create({"service.name": SERVICE})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    _installed = Telemetry(tracer_provider, OtlpLogExport(logger_provider))
    return _installed


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
