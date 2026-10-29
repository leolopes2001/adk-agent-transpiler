import io
import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import mcp.shared.session
import pytest
import structlog

from lab_common import logging as lab_logging
from ocr_mcp.logs import REDACTED, LibraryLogFilter, configure_logging

pytestmark = pytest.mark.unit

PAYLOAD = "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZ0FBQUFF"
MCP_FILE = os.path.abspath(mcp.shared.session.__file__)


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    configure_logging("DEBUG", stream=buffer)
    yield buffer
    if lab_logging._handler is not None:
        logging.getLogger().removeHandler(lab_logging._handler)
        lab_logging._handler = None
    logging.getLogger().setLevel(logging.WARNING)
    structlog.reset_defaults()


def records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def record(
    name: str, msg: str, *, args: tuple[object, ...] = (), pathname: str = __file__, **extra: Any
) -> logging.LogRecord:
    rec = logging.LogRecord(name, logging.ERROR, pathname, 1, msg, args, None)
    rec.__dict__.update(extra)
    return rec


class TestLibraryRecords:
    @pytest.mark.parametrize("logger", ["presidio-analyzer", "presidio_analyzer.nlp", "spacy"])
    def test_ner_library_messages_are_redacted(self, stream: io.StringIO, logger: str) -> None:
        try:
            raise ValueError("Paciente: Maria Fictícia da Silva")
        except ValueError:
            logging.getLogger(logger).exception("Entity failed: Maria Fictícia da Silva")

        [rec] = records(stream)
        assert rec["event"] == f"Entity failed: {REDACTED}"
        assert rec["exception"] == REDACTED
        assert "Maria" not in stream.getvalue()

    def test_exception_keeps_only_its_type(self, stream: io.StringIO) -> None:
        try:
            raise ValueError(f"input_value='{PAYLOAD}'")
        except ValueError:
            logging.getLogger("mcp.server.sse").exception("Failed to parse message")

        [rec] = records(stream)
        assert rec["event"] == "Failed to parse message"
        assert rec["exception_type"] == "ValueError"
        assert rec["exception"] == REDACTED
        assert PAYLOAD not in stream.getvalue()

    @pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.ERROR])
    def test_text_after_the_first_colon_is_redacted(self, stream: io.StringIO, level: int) -> None:
        logging.getLogger("mcp.server.lowlevel.server").log(
            level, f"Received exception from stream: 1 validation error {PAYLOAD}"
        )

        assert records(stream)[0]["event"] == f"Received exception from stream: {REDACTED}"

    def test_formatting_arguments_are_redacted_too(self, stream: io.StringIO) -> None:
        logging.getLogger("uvicorn.error").error("Invalid request: %s", PAYLOAD)
        assert records(stream)[0]["event"] == f"Invalid request: {REDACTED}"

    def test_root_logger_calls_from_library_code_are_caught(self) -> None:
        # mcp.shared.session logs rejected messages with logging.warning(...).
        rec = record(
            "root", f"Failed to validate notification: x. Message was: {PAYLOAD}", pathname=MCP_FILE
        )

        LibraryLogFilter().filter(rec)

        assert rec.getMessage() == f"Failed to validate notification: {REDACTED}"

    def test_message_formatted_by_an_earlier_handler_is_dropped(self) -> None:
        rec = record("mcp.server.lowlevel.server", f"Received exception from stream: {PAYLOAD}")
        logging.Formatter().format(rec)  # sets rec.message, as pytest's handler does

        LibraryLogFilter().filter(rec)

        assert "message" not in rec.__dict__
        assert PAYLOAD not in rec.getMessage()

    def test_uvicorn_color_message_is_removed(self, stream: io.StringIO) -> None:
        logging.getLogger("uvicorn.error").info(
            "Started server process [%d]", 42, extra={"color_message": "\x1b[36m%d\x1b[0m"}
        )

        [rec] = records(stream)
        assert rec["event"] == "Started server process [42]"
        assert "color_message" not in rec

    def test_messages_without_a_colon_are_kept(self, stream: io.StringIO) -> None:
        logging.getLogger("uvicorn.error").info(
            "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)", "http", "127.0.0.1", 8101
        )
        assert records(stream)[0]["event"] == (
            "Uvicorn running on http://127.0.0.1:8101 (Press CTRL+C to quit)"
        )


class TestOtherRecords:
    def test_own_records_are_untouched(self, stream: io.StringIO) -> None:
        lab_logging.get_logger("ocr_mcp.server").info("ocr_done", note="a: b")
        logging.getLogger("mcpserver_lookalike").error("detail: kept")

        assert [(r["event"], r.get("note")) for r in records(stream)] == [
            ("ocr_done", "a: b"),
            ("detail: kept", None),
        ]

    def test_own_exceptions_keep_their_traceback(self, stream: io.StringIO) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logging.getLogger("ocr_mcp.server").exception("failed")

        assert "RuntimeError: boom" in records(stream)[0]["exception"]

    def test_filter_never_drops_records(self) -> None:
        assert LibraryLogFilter().filter(record("mcp", "x: y"))
        assert LibraryLogFilter().filter(record("other", "x: y"))
