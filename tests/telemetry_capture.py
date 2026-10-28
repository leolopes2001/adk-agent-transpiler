"""In-memory export of what `lab_common.telemetry` would send to the collector.

`install` sets up the process' global tracer provider through the same
`setup` the services use, with in-memory exporters; OpenTelemetry accepts one
global provider per process, so the `exported` fixture (tests/conftest.py)
installs it once per session and empties it around each test.
"""

from dataclasses import dataclass

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lab_common import telemetry
from lab_common.telemetry import Telemetry


@dataclass
class Exported:
    telemetry: Telemetry
    span_exporter: InMemorySpanExporter
    log_exporter: InMemoryLogRecordExporter

    def spans(self) -> list[ReadableSpan]:
        self.telemetry.tracer_provider.force_flush()
        return list(self.span_exporter.get_finished_spans())

    def logs(self) -> list[ReadableLogRecord]:
        """Records that went through `telemetry.log_export` (pass it to the logging setup)."""
        self.telemetry.logger_provider.force_flush()
        return list(self.log_exporter.get_finished_logs())

    def clear(self) -> None:
        self.telemetry.tracer_provider.force_flush()
        self.telemetry.logger_provider.force_flush()
        self.span_exporter.clear()
        self.log_exporter.clear()


def install(service: str = "test") -> Exported:
    span_exporter = InMemorySpanExporter()
    log_exporter = InMemoryLogRecordExporter()
    installed = telemetry.setup(service, span_exporter=span_exporter, log_exporter=log_exporter)
    assert installed is not None
    return Exported(installed, span_exporter, log_exporter)
