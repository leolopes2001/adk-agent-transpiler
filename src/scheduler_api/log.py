"""Structured JSON logging for the API.

The API is deployed as an external service and imports no other project
package, so it carries its own small version of the shared logging setup: one
JSON object per line on stderr with `ts`, `level`, `service` and `event`, plus
whatever is bound to the context (`request_id`). Records from the standard
`logging` module (uvicorn, sqlalchemy, alembic) go through the same pipeline;
uvicorn's own text handlers, installed before the app factory runs, are
removed so its records reach the JSON handler too.

Request and response bodies are never logged; `_drop_forbidden_keys` removes
them if a call ever passes one by mistake.

`export` (see `scheduler_api.telemetry`) receives every rendered line with its
record: what is exported is exactly what is written to stderr, from INFO up
whatever the configured level.
"""

import logging
import sys
from collections.abc import Callable
from typing import TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

SERVICE = "scheduler-api"

_FORBIDDEN_KEYS = frozenset({"body", "request_body", "response_body"})
# uvicorn repeats each message with ANSI colors in `color_message` for its own formatter.
_DROPPED_KEYS = _FORBIDDEN_KEYS | {"color_message"}
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

type LogExport = Callable[[logging.LogRecord, str], None]
"""Receives a record and its rendered JSON line."""

_handler: logging.Handler | None = None


class _JsonLinesHandler(logging.StreamHandler):
    """Writes records at `stream_level` and up; hands records at INFO and up, with the
    same rendered line, to `export`."""

    def __init__(self, stream: TextIO, stream_level: int, export: LogExport | None) -> None:
        super().__init__(stream)
        self._stream_level = stream_level
        self._export = export
        self._export_level = min(stream_level, logging.INFO)
        self.setLevel(self._export_level if export else stream_level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            if record.levelno >= self._stream_level:
                self.stream.write(line + self.terminator)
                self.flush()
            if self._export is not None and record.levelno >= self._export_level:
                self._export(record, line)
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - logging must never break the caller
            self.handleError(record)


def _add_service(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict["service"] = SERVICE
    return event_dict


def _drop_forbidden_keys(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    for key in _DROPPED_KEYS.intersection(event_dict):
        del event_dict[key]
    return event_dict


def configure_logging(
    level: str = "INFO", *, stream: TextIO | None = None, export: LogExport | None = None
) -> None:
    """Configure structlog and the stdlib root logger; calling it again replaces the setup."""
    global _handler  # noqa: PLW0603 - one handler per process, replaced on reconfigure

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _add_service,
        structlog.processors.format_exc_info,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *shared],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _drop_forbidden_keys,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = _JsonLinesHandler(
        stream or sys.stderr, logging.getLevelNamesMapping()[level.upper()], export
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    root.addHandler(handler)
    root.setLevel(handler.level)
    _handler = handler
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger
