"""In-memory export of what `lab_common.telemetry` would send to the collector.

`install` sets up the process' global tracer provider through the same
`setup` the services use, with in-memory exporters; OpenTelemetry accepts one
global provider per process, so the `exported` fixture (tests/conftest.py)
installs it once per session and empties it around each test.
"""

from dataclasses import dataclass

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lab_common import telemetry
from lab_common.telemetry import Telemetry


@dataclass
class Exported:
    telemetry: Telemetry
    span_exporter: InMemorySpanExporter

    def spans(self) -> list[ReadableSpan]:
        self.telemetry.tracer_provider.force_flush()
        return list(self.span_exporter.get_finished_spans())

    def clear(self) -> None:
        self.telemetry.tracer_provider.force_flush()
        self.span_exporter.clear()


def install(service: str = "test") -> Exported:
    span_exporter = InMemorySpanExporter()
    installed = telemetry.setup(service, span_exporter=span_exporter)
    assert installed is not None
    return Exported(installed, span_exporter)
