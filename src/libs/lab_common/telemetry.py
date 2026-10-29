import json
import logging
import os
import socket
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import AttributeValue

__all__ = [
    "ALLOWED_ATTRIBUTES",
    "ENDPOINT_ENV",
    "OtlpLogExport",
    "SafeSpanExporter",
    "Status",
    "StatusCode",
    "Telemetry",
    "annotate",
    "collector_endpoint",
    "context_from_meta",
    "fail",
    "inject_headers",
    "request_meta",
    "resource",
    "safe_span",
    "server_span",
    "setup",
    "trace_meta",
    "tracer",
]

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
PROBE_TIMEOUT_S = 0.5
# A collector whose name resolves may still be starting.
STARTUP_WAIT_S = 10.0
_PROBE_INTERVAL_S = 0.25

ATTRIBUTE_PREFIX = "lab."
# Ids, names, models, counts and codes only: never prompts, responses or tool payloads.
ALLOWED_ATTRIBUTES = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.workflow.name",
        "gen_ai.workflow.nested",
        "gen_ai.conversation.id",
        "gen_ai.system",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.tool.name",
        "error.type",
        "http.request.method",
        "http.route",
        "http.response.status_code",
        "server.address",
        "server.port",
        "mcp.method.name",
    }
)
ALLOWED_EVENT_ATTRIBUTES = frozenset({"exception.type"})

# W3C traceparent only: baggage is never propagated.
_PROPAGATOR = TraceContextTextMapPropagator()

_SEVERITY_BY_LEVEL = {
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}
# The SDK logs its own export failures; exporting those would feed the failure.
_NOT_EXPORTED_LOGGERS = ("opentelemetry",)


class SafeSpanExporter(SpanExporter):
    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([safe_span(span) for span in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


class OtlpLogExport:
    def __init__(self, provider: LoggerProvider) -> None:
        self._logger = provider.get_logger("lab_common.logging")
        self._closed = False

    def __call__(self, record: logging.LogRecord, line: str) -> None:
        if self._closed or record.name.startswith(_NOT_EXPORTED_LOGGERS):
            return

        self._logger.emit(
            LogRecord(
                timestamp=int(record.created * 1e9),
                context=context_api.get_current(),
                severity_text=record.levelname,
                severity_number=_SEVERITY_BY_LEVEL.get(
                    record.levelname, SeverityNumber.UNSPECIFIED
                ),
                body=line,
                attributes=_scalar_fields(line),
            )
        )

    def close(self) -> None:
        self._closed = True


@dataclass
class Telemetry:
    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    log_export: OtlpLogExport

    def shutdown(self) -> None:
        self.log_export.close()
        self.logger_provider.shutdown()
        self.tracer_provider.shutdown()


def collector_endpoint(wait_s: float = STARTUP_WAIT_S) -> str | None:
    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    url = urlsplit(endpoint)
    if not endpoint or url.hostname is None:
        return None

    port = url.port or (443 if url.scheme == "https" else 80)
    # Compose always sets the variable; the name only resolves with the observability profile.
    if not _resolves(url.hostname, port):
        return None

    if not _accepts_connection(url.hostname, port, wait_s):
        return None

    return endpoint.rstrip("/")


def allowed(key: str) -> bool:
    return key.startswith(ATTRIBUTE_PREFIX) or key in ALLOWED_ATTRIBUTES


def safe_span(span: ReadableSpan) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_filter_attributes(span.attributes, allowed),
        events=[
            Event(
                event.name,
                _filter_attributes(event.attributes, ALLOWED_EVENT_ATTRIBUTES.__contains__),
                timestamp=event.timestamp,
            )
            for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        # No description: an error message can quote its input.
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def resource(service: str, instance_id: str | None = None) -> Resource:
    attributes = {"service.name": service}
    if instance_id is not None:
        attributes["service.instance.id"] = instance_id

    return Resource.create(attributes)


def setup(
    service: str,
    *,
    instance_id: str | None = None,
    span_exporter: SpanExporter | None = None,
    log_exporter: LogRecordExporter | None = None,
) -> Telemetry | None:
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

    service_resource = resource(service, instance_id)
    tracer_provider = TracerProvider(resource=service_resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(SafeSpanExporter(span_exporter)))
    trace.set_tracer_provider(tracer_provider)

    # Never the global provider, so library log events (ADK's gen_ai.*) never leave the process.
    logger_provider = LoggerProvider(resource=service_resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    return Telemetry(tracer_provider, logger_provider, OtlpLogExport(logger_provider))


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def trace_meta() -> dict[str, str]:
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier


def inject_headers(headers: dict[str, str]) -> dict[str, str]:
    _PROPAGATOR.inject(headers)
    return headers


def context_from_meta(meta: Mapping[str, Any] | None) -> context_api.Context:
    carrier = {k: v for k, v in (meta or {}).items() if isinstance(v, str)}
    return _PROPAGATOR.extract(carrier)


def annotate(**values: AttributeValue | None) -> None:
    span = trace.get_current_span()
    for key, value in values.items():
        if value is not None:
            span.set_attribute(f"{ATTRIBUTE_PREFIX}{key}", value)


def fail(code: str, span: Span | None = None) -> None:
    # A code, never a message: messages can quote the input.
    span = span or trace.get_current_span()
    span.set_attribute(f"{ATTRIBUTE_PREFIX}error_code", code)
    span.set_status(Status(StatusCode.ERROR))


def request_meta(ctx: Any) -> Mapping[str, Any] | None:
    try:
        meta = ctx.request_context.meta
    except (AttributeError, LookupError, ValueError):
        return None

    return meta.model_dump() if meta is not None else None


@contextmanager
def server_span(
    name: str, meta: Mapping[str, Any] | None, **values: AttributeValue | None
) -> Iterator[Span]:
    with tracer(__name__).start_as_current_span(
        name,
        context=context_from_meta(meta),
        kind=SpanKind.SERVER,
        record_exception=False,
    ) as span:
        annotate(**values)
        yield span


def _resolves(host: str, port: int) -> bool:
    try:
        socket.getaddrinfo(host, port)
    except OSError:
        return False

    return True


def _accepts_connection(host: str, port: int, wait_s: float) -> bool:
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False

            time.sleep(_PROBE_INTERVAL_S)


def _filter_attributes(
    attributes: Mapping[str, AttributeValue] | None, is_allowed: Callable[[str], bool]
) -> dict[str, AttributeValue]:
    return {key: value for key, value in (attributes or {}).items() if is_allowed(key)}


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
