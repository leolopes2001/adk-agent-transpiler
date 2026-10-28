"""Optional OpenTelemetry traces and logs for the agent runtime and the MCP servers.

Export is off unless OTEL_EXPORTER_OTLP_ENDPOINT is set and a collector
accepts a connection there when the process starts (`collector_endpoint`).
Docker Compose always sets the variable, and the name only resolves when the
`observability` profile is up, so the default stack exports nothing and
behaves as before. With export off, no provider is installed and every span
is the API's no-op span.

Spans carry ids, names, counts, codes, statuses and durations only. Every
span goes through `SafeSpanExporter` before it leaves the process: attributes
outside `ALLOWED_ATTRIBUTES` are dropped (prompts, model responses and tool
payloads that libraries such as ADK may record), event attributes are reduced
to the exception type and status descriptions are removed, since an error
message can quote its input.

Logs are exported by `OtlpLogExport`, which `lab_common.logging` hands each
rendered line after redaction: the body of every exported record is the
exact JSON line written to stderr, its attributes are that line's top-level
scalar fields and it carries the trace and span id of the span it was logged
in. The logger provider is private to this export (never the global one), so
log events emitted by libraries such as ADK's `gen_ai.*` events never leave
the process; the SDK's own records are not exported either.

Context crosses process boundaries as W3C `traceparent`: in the `_meta` of an
MCP request (`trace_meta`, `context_from_meta`) and in HTTP headers. Baggage is
never propagated.

Requires the `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`
packages, which the agent, OCR and RAG extras install.
"""

import json
import logging
import os
import socket
from collections.abc import Iterator, Mapping, Sequence
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
"""How long the start-up check waits for the collector; an unknown name fails at once."""

ATTRIBUTE_PREFIX = "lab."
"""Attributes this project sets: `lab.run_id`, `lab.node`, `lab.route`, counts, codes."""
ALLOWED_ATTRIBUTES = frozenset(
    {
        # ADK spans: operation, names, model and token counts (never content).
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
        # HTTP and MCP calls.
        "http.request.method",
        "http.route",
        "http.response.status_code",
        "server.address",
        "server.port",
        "mcp.method.name",
    }
)
ALLOWED_EVENT_ATTRIBUTES = frozenset({"exception.type"})

_PROPAGATOR = TraceContextTextMapPropagator()


def collector_endpoint() -> str | None:
    """OTEL_EXPORTER_OTLP_ENDPOINT when a collector listens there, else None."""
    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    if not endpoint:
        return None
    url = urlsplit(endpoint)
    if url.hostname is None:
        return None
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        with socket.create_connection((url.hostname, port), timeout=PROBE_TIMEOUT_S):
            pass
    except OSError:
        return None
    return endpoint.rstrip("/")


def allowed(key: str) -> bool:
    return key.startswith(ATTRIBUTE_PREFIX) or key in ALLOWED_ATTRIBUTES


class SafeSpanExporter(SpanExporter):
    """Exports copies of the spans with only the allowed attributes (see module docstring)."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([safe_span(span) for span in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


def _attributes(
    attributes: Mapping[str, AttributeValue] | None, keep: frozenset[str] | None = None
) -> dict[str, AttributeValue]:
    return {
        key: value
        for key, value in (attributes or {}).items()
        if (key in keep if keep is not None else allowed(key))
    }


def safe_span(span: ReadableSpan) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_attributes(span.attributes),
        events=[
            Event(
                event.name,
                _attributes(event.attributes, ALLOWED_EVENT_ATTRIBUTES),
                timestamp=event.timestamp,
            )
            for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


_SEVERITY = {
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}
_NOT_EXPORTED_LOGGERS = ("opentelemetry",)
"""The SDK logs its own export failures; exporting those would feed the failure."""


class OtlpLogExport:
    """`lab_common.logging.LogExport` that sends each rendered line as an OTLP log record."""

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
                severity_number=_SEVERITY.get(record.levelname, SeverityNumber.UNSPECIFIED),
                body=line,
                attributes=_scalar_fields(line),
            )
        )

    def close(self) -> None:
        """Later records are not exported (the provider is being shut down)."""
        self._closed = True


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


@dataclass
class Telemetry:
    """What `setup` installed; `shutdown` flushes and stops the exporters."""

    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    log_export: OtlpLogExport

    def shutdown(self) -> None:
        self.log_export.close()
        self.logger_provider.shutdown()
        self.tracer_provider.shutdown()


def resource(service: str, instance_id: str | None = None) -> Resource:
    """`service.name`, plus `service.instance.id` for a process that is one run (the CLI)."""
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
    """Install the global tracer provider and a log export; None (nothing) when export is off.

    Without exporters, spans and logs go to the collector at
    OTEL_EXPORTER_OTLP_ENDPOINT, if one is there. Tests pass in-memory exporters.
    Pass `Telemetry.log_export` to `lab_common.logging.configure(export=...)`.
    """
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
    logger_provider = LoggerProvider(resource=service_resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    return Telemetry(tracer_provider, logger_provider, OtlpLogExport(logger_provider))


def tracer(name: str) -> trace.Tracer:
    """A tracer of the global provider (a no-op until `setup` installs one)."""
    return trace.get_tracer(name)


def trace_meta() -> dict[str, str]:
    """`traceparent` of the current span, for the `_meta` of an MCP request (empty if none)."""
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier


def inject_headers(headers: dict[str, str]) -> dict[str, str]:
    """`headers` plus the `traceparent` of the current span, if any."""
    _PROPAGATOR.inject(headers)
    return headers


def context_from_meta(meta: Mapping[str, Any] | None) -> context_api.Context:
    """Parent context carried by an MCP request's `_meta`; the current one if it has none."""
    carrier = {k: v for k, v in (meta or {}).items() if isinstance(v, str)}
    return _PROPAGATOR.extract(carrier)


def annotate(**values: AttributeValue | None) -> None:
    """Set `lab.<name>` attributes on the current span (None values are skipped)."""
    span = trace.get_current_span()
    for key, value in values.items():
        if value is not None:
            span.set_attribute(f"{ATTRIBUTE_PREFIX}{key}", value)


def fail(code: str, span: Span | None = None) -> None:
    """Mark `span` (default: the current one) failed with an error code, never a message."""
    span = span or trace.get_current_span()
    span.set_attribute(f"{ATTRIBUTE_PREFIX}error_code", code)
    span.set_status(Status(StatusCode.ERROR))


def request_meta(ctx: Any) -> Mapping[str, Any] | None:
    """`_meta` of the MCP request a FastMCP `Context` belongs to; None outside a request."""
    try:
        meta = ctx.request_context.meta
    except (AttributeError, LookupError, ValueError):
        return None
    return meta.model_dump() if meta is not None else None


@contextmanager
def server_span(
    name: str, meta: Mapping[str, Any] | None, **values: AttributeValue | None
) -> Iterator[Span]:
    """SERVER span of a request handled here, child of the caller's span from `meta`."""
    with tracer(__name__).start_as_current_span(
        name,
        context=context_from_meta(meta),
        kind=SpanKind.SERVER,
        record_exception=False,
    ) as span:
        annotate(**values)
        yield span
