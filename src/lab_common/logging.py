"""Structured JSON logging shared by the agent runtime and the MCP servers.

Every record is one JSON object on stderr with at least `ts`, `level`,
`service` and `event`. Correlation fields (`run_id`, `request_id`, `node`, ...)
are added by binding them with `structlog.contextvars.bind_contextvars` or by
passing them as keyword arguments to the log call.

Records emitted through the standard `logging` module (uvicorn, mcp, httpx)
go through the same pipeline, so they are rendered as JSON too and pass
through the same final processors, including their `extra=` fields.

PII redaction is on by default and runs last, after any `final_processors`:
forbidden keys (`raw_text`, `image_base64`, ...) are dropped at any depth,
values matching a high-precision detector (valid CPF, CNS, e-mail, phone) are
replaced with `[REDACTED:<TYPE>]` and long strings are truncated. It is a
safety net; code must still log counts, hashes and codes, never document text.

stderr is used so a CLI can keep stdout for its own output.
"""

import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from lab_common.pii import PiiDetector, high_precision_detectors
from lab_common.pii.redactor import detect_all

_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

FORBIDDEN_KEYS = frozenset({"raw_text", "text_raw", "image_base64", "image_bytes", "body"})
MAX_STRING_CHARS = 500
_TRUNCATED = "…[truncated]"

_handler: logging.Handler | None = None


class _AddService:
    def __init__(self, service: str) -> None:
        self._service = service

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = self._service
        return event_dict


class RedactPii:
    """Final processor that keeps PII out of log records (see module docstring)."""

    def __init__(self, detectors: Sequence[PiiDetector] | None = None) -> None:
        self._detectors = tuple(detectors or high_precision_detectors())

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        return self._mapping(event_dict)

    def _mapping(self, mapping: Mapping[Any, Any]) -> dict[str, object]:
        return {
            self._string(str(key)): self._value(value)
            for key, value in mapping.items()
            if key not in FORBIDDEN_KEYS
        }

    def _value(self, value: object) -> object:
        if value is None or isinstance(value, bool | float):
            return value
        if isinstance(value, int):
            text = str(value)
            redacted = self._string(text)
            return value if redacted == text else redacted
        if isinstance(value, Mapping):
            return self._mapping(value)
        if isinstance(value, list | tuple | set | frozenset):
            return [self._value(item) for item in value]
        if isinstance(value, bytes | bytearray | memoryview):
            return "[REDACTED:BYTES]"
        # Anything else would be rendered through repr() by the JSON renderer,
        # so it is converted here, where it can still be checked.
        return self._string(value if isinstance(value, str) else str(value))

    def _string(self, text: str) -> str:
        spans = detect_all(self._detectors, text, min_score=0.0)
        if spans:
            pieces = []
            cursor = 0
            for span in spans:
                pieces.append(text[cursor : span.start])
                pieces.append(f"[REDACTED:{span.entity}]")
                cursor = span.end
            pieces.append(text[cursor:])
            text = "".join(pieces)
        if len(text) > MAX_STRING_CHARS:
            text = text[:MAX_STRING_CHARS] + _TRUNCATED
        return text


def configure(  # noqa: PLR0913 - all but service and level are keyword-only options
    service: str,
    level: str = "INFO",
    *,
    final_processors: Sequence[Processor] = (),
    stream: TextIO | None = None,
    redact_pii: bool = True,
    filters: Sequence[logging.Filter] = (),
) -> None:
    """Configure structlog and the stdlib root logger to emit JSON lines.

    `final_processors` run after all standard fields are set and right before
    rendering, so they see the complete record. PII redaction (`RedactPii`)
    runs after them unless `redact_pii` is False, which is meant only for
    tests of the logging pipeline itself. `filters` are attached to the
    handler, so they see every stdlib record (with its logger name) before it
    is formatted. Calling `configure` again replaces
    the previous configuration.
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
            *([RedactPii()] if redact_pii else []),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    for log_filter in filters:
        handler.addFilter(log_filter)

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
