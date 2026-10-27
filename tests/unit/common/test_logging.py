import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from structlog.typing import EventDict, WrappedLogger

from lab_common import logging as lab_logging

pytestmark = pytest.mark.unit


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


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_emits_one_json_object_per_event_with_standard_fields(stream: io.StringIO) -> None:
    lab_logging.configure("agent", stream=stream)

    lab_logging.get_logger().info("run_started", run_id="r-1", attempt=1)

    [record] = records(stream)
    assert record["event"] == "run_started"
    assert record["level"] == "info"
    assert record["service"] == "agent"
    assert record["run_id"] == "r-1"
    assert record["attempt"] == 1
    assert datetime.fromisoformat(record["ts"]).utcoffset() == UTC.utcoffset(None)


def test_filters_below_configured_level(stream: io.StringIO) -> None:
    lab_logging.configure("agent", "warning", stream=stream)
    log = lab_logging.get_logger()

    log.info("hidden")
    log.warning("shown")

    assert [r["event"] for r in records(stream)] == ["shown"]


def test_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        lab_logging.configure("agent", "verbose")


def test_includes_context_bound_variables(stream: io.StringIO) -> None:
    lab_logging.configure("agent", stream=stream)
    structlog.contextvars.bind_contextvars(run_id="r-2", node="ocr")

    lab_logging.get_logger().info("node_started")

    [record] = records(stream)
    assert record["run_id"] == "r-2"
    assert record["node"] == "ocr"


def test_stdlib_records_are_rendered_as_json(stream: io.StringIO) -> None:
    lab_logging.configure("rag-mcp", stream=stream)

    logging.getLogger("uvicorn.error").warning("server %s", "started")

    [record] = records(stream)
    assert record["event"] == "server started"
    assert record["level"] == "warning"
    assert record["service"] == "rag-mcp"
    assert "ts" in record


def test_renders_exception_as_string(stream: io.StringIO) -> None:
    lab_logging.configure("agent", stream=stream)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        lab_logging.get_logger().exception("node_failed")

    [record] = records(stream)
    assert "RuntimeError: boom" in record["exception"]


def test_keeps_non_ascii_characters_readable(stream: io.StringIO) -> None:
    lab_logging.configure("agent", stream=stream)

    lab_logging.get_logger().info("catalog_loaded", sample="Ácido úrico")

    assert "Ácido úrico" in stream.getvalue()


@pytest.mark.parametrize("emit", ["structlog", "stdlib"])
def test_final_processors_see_complete_record_and_can_rewrite_it(
    stream: io.StringIO, emit: str
) -> None:
    seen: list[EventDict] = []

    def drop_raw_text(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        seen.append(dict(event_dict))
        event_dict.pop("raw_text", None)
        return event_dict

    lab_logging.configure("ocr-mcp", stream=stream, final_processors=[drop_raw_text])
    if emit == "structlog":
        lab_logging.get_logger().info("ocr_done", raw_text="x")
    else:
        logging.getLogger("mcp").info("ocr_done", extra={"raw_text": "x"})

    [record] = records(stream)
    assert {"ts", "level", "service", "event", "raw_text"} <= seen[0].keys()
    assert "raw_text" not in record
    assert record["event"] == "ocr_done"


def test_reconfiguring_replaces_previous_handler(stream: io.StringIO) -> None:
    old = io.StringIO()
    lab_logging.configure("agent", stream=old)
    lab_logging.configure("agent", stream=stream)

    lab_logging.get_logger().info("run_finished")

    assert old.getvalue() == ""
    assert len(records(stream)) == 1


def test_logger_created_before_configure_uses_new_configuration(stream: io.StringIO) -> None:
    log = lab_logging.get_logger("early")
    lab_logging.configure("agent", stream=stream)

    log.info("late_event")

    assert records(stream)[0]["service"] == "agent"
