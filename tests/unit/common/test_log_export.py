"""Log export: what reaches the collector is exactly the redacted line on stderr."""

import io
import logging
from collections.abc import Iterator

import pytest
import structlog

from lab_common import logging as lab_logging
from lab_common import telemetry
from telemetry_capture import Exported

pytestmark = pytest.mark.unit

CPF = "202.393.613-69"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    if lab_logging._handler is not None:
        root.removeHandler(lab_logging._handler)
        lab_logging._handler = None
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def test_exported_line_is_the_redacted_stderr_line(exported: Exported) -> None:
    stream = io.StringIO()
    lab_logging.configure("agent", stream=stream, export=exported.telemetry.log_export)

    lab_logging.get_logger("t").warning("lookup", cpf=CPF, raw_text="Paciente: Maria")

    [line] = stream.getvalue().splitlines()
    [record] = exported.logs()
    assert record.log_record.body == line
    assert CPF not in line
    assert "Maria" not in line
    assert record.log_record.severity_text == "WARNING"


def test_exports_info_while_the_terminal_shows_errors_only(exported: Exported) -> None:
    stream = io.StringIO()
    lab_logging.configure("agent", "ERROR", stream=stream, export=exported.telemetry.log_export)
    log = lab_logging.get_logger("t")

    log.debug("not_exported")
    log.info("run_started")
    log.error("run_failed", error_code="MCP_TIMEOUT")

    assert [line for line in stream.getvalue().splitlines() if "run_started" in line] == []
    events = [r.log_record.attributes["event"] for r in exported.logs()]  # type: ignore[index]
    assert events == ["run_started", "run_failed"]


def test_records_carry_the_trace_and_span_they_were_logged_in(exported: Exported) -> None:
    lab_logging.configure("agent", stream=io.StringIO(), export=exported.telemetry.log_export)
    structlog.contextvars.bind_contextvars(run_id="r-1")

    with telemetry.tracer(__name__).start_as_current_span("invoke_node ocr") as span:
        lab_logging.get_logger("t").info("ocr_result", lines=20, counts={"CPF": 1})

    [record] = exported.logs()
    context = span.get_span_context()
    assert (record.log_record.trace_id, record.log_record.span_id) == (
        context.trace_id,
        context.span_id,
    )
    attributes = dict(record.log_record.attributes or {})
    assert {k: attributes[k] for k in ("event", "run_id", "lines", "service", "level")} == {
        "event": "ocr_result",
        "run_id": "r-1",
        "lines": 20,
        "service": "agent",
        "level": "info",
    }
    # Nested fields stay in the body only; the timestamp is the record's own.
    assert "counts" not in attributes
    assert "ts" not in attributes


def test_sdk_records_and_records_after_shutdown_are_not_exported(exported: Exported) -> None:
    provider = exported.telemetry.logger_provider
    export = telemetry.OtlpLogExport(provider)
    lab_logging.configure("agent", stream=io.StringIO(), export=export)

    logging.getLogger("opentelemetry.sdk._logs").warning("Failed to export logs batch")
    export.close()
    lab_logging.get_logger("t").info("after_shutdown")

    assert exported.logs() == []


def test_without_export_the_levels_are_unchanged() -> None:
    lab_logging.configure("agent", "ERROR", stream=io.StringIO())

    assert logging.getLogger().level == logging.ERROR
