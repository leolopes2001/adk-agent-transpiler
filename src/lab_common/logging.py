"""Structured JSON logging shared by the agent runtime and the MCP servers.

Every record is one JSON object on stderr with at least `ts`, `level`,
`service` and `event`. Correlation fields (`run_id`, `request_id`, `node`, ...)
are added by binding them with `structlog.contextvars.bind_contextvars` or by
passing them as keyword arguments to the log call.

Records emitted through the standard `logging` module (uvicorn, mcp, httpx)
go through the same pipeline, so they are rendered as JSON too and pass
through the same final processors, including their `extra=` fields.

stderr is used so a CLI can keep stdout for its own output.
"""

import logging
import sys
from collections.abc import Sequence
from typing import TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_handler: logging.Handler | None = None


class _AddService:
    def __init__(self, service: str) -> None:
        self._service = service

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = self._service
        return event_dict


def configure(
    service: str,
    level: str = "INFO",
    *,
    final_processors: Sequence[Processor] = (),
    stream: TextIO | None = None,
) -> None:
    """Configure structlog and the stdlib root logger to emit JSON lines.

    `final_processors` run after all standard fields are set and right before
    rendering, so they see the complete record; this is where log redaction
    plugs in. Calling `configure` again replaces the previous configuration.
    """
    global _handler  # noqa: PLW0603 - one handler per process, replaced on reconfigure

    level_name = level.upper()
    if level_name not in _LEVELS:
        raise ValueError(f"unknown log level {level!r}; expected one of {sorted(_LEVELS)}")

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _AddService(service),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        # ExtraAdder keeps `extra=` fields from stdlib calls, so they reach the
        # final processors instead of bypassing them.
        foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *shared],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *final_processors,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    root.addHandler(handler)
    root.setLevel(level_name)
    _handler = handler

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Not cached, so loggers created at import time pick up a later configure().
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger
