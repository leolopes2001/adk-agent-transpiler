"""Structured JSON logging for the API.

The API is deployed as an external service and imports no other project
package, so it carries its own small version of the shared logging setup: one
JSON object per line on stderr with `ts`, `level`, `service` and `event`, plus
whatever is bound to the context (`request_id`). Records from the standard
`logging` module (uvicorn, sqlalchemy) go through the same pipeline.

Request and response bodies are never logged; `_drop_forbidden_keys` removes
them if a call ever passes one by mistake.
"""

import logging
import sys
from typing import TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

SERVICE = "scheduler-api"

_FORBIDDEN_KEYS = frozenset({"body", "request_body", "response_body"})

_handler: logging.Handler | None = None


def _add_service(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict["service"] = SERVICE
    return event_dict


def _drop_forbidden_keys(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    for key in _FORBIDDEN_KEYS.intersection(event_dict):
        del event_dict[key]
    return event_dict


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
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
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    root.addHandler(handler)
    root.setLevel(level.upper())
    _handler = handler

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
