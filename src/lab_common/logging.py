import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from lab_common.pii import PiiDetector, high_precision_detectors
from lab_common.pii.redactor import detect_all

_LEVEL_NAMES = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

FORBIDDEN_KEYS = frozenset({"raw_text", "text_raw", "image_base64", "image_bytes", "body"})
MAX_STRING_CHARS = 500
_TRUNCATED = "…[truncated]"

type LogExport = Callable[[logging.LogRecord, str], None]
"""Receives each record with its rendered line, already redacted."""

_handler: logging.Handler | None = None


class _AddService:
    def __init__(self, service: str) -> None:
        self._service = service

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = self._service
        return event_dict


class _JsonLinesHandler(logging.StreamHandler):  # type: ignore[type-arg]
    def __init__(
        self,
        stream: TextIO,
        stream_level: int,
        export: LogExport | None = None,
        export_level: int = logging.INFO,
    ) -> None:
        super().__init__(stream)
        self._stream_level = stream_level
        self._export = export
        self._export_level = export_level
        self.setLevel(min(stream_level, export_level) if export else stream_level)

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


class RedactPii:
    def __init__(self, detectors: Sequence[PiiDetector] | None = None) -> None:
        self._detectors = tuple(detectors or high_precision_detectors())

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        return self._redact_mapping(event_dict)

    def _redact_mapping(self, mapping: Mapping[Any, Any]) -> dict[str, object]:
        return {
            self._redact_string(str(key)): self._redact_value(value)
            for key, value in mapping.items()
            if key not in FORBIDDEN_KEYS
        }

    def _redact_value(self, value: object) -> object:
        if value is None or isinstance(value, bool | float):
            return value

        if isinstance(value, int):
            text = str(value)
            redacted = self._redact_string(text)
            return value if redacted == text else redacted

        if isinstance(value, Mapping):
            return self._redact_mapping(value)

        if isinstance(value, list | tuple | set | frozenset):
            return [self._redact_value(item) for item in value]

        if isinstance(value, bytes | bytearray | memoryview):
            return "[REDACTED:BYTES]"

        # Converted here, where it can still be checked; the JSON renderer would use repr().
        return self._redact_string(value if isinstance(value, str) else str(value))

    def _redact_string(self, text: str) -> str:
        masked = self._mask_spans(text)
        if len(masked) > MAX_STRING_CHARS:
            return masked[:MAX_STRING_CHARS] + _TRUNCATED

        return masked

    def _mask_spans(self, text: str) -> str:
        pieces = []
        cursor = 0
        for span in detect_all(self._detectors, text, min_score=0.0):
            pieces.append(text[cursor : span.start])
            pieces.append(f"[REDACTED:{span.entity}]")
            cursor = span.end

        pieces.append(text[cursor:])
        return "".join(pieces)


def configure(  # noqa: PLR0913 - all but service and level are keyword-only options
    service: str,
    level: str = "INFO",
    *,
    final_processors: Sequence[Processor] = (),
    stream: TextIO | None = None,
    redact_pii: bool = True,
    filters: Sequence[logging.Filter] = (),
    export: LogExport | None = None,
    export_level: str = "INFO",
) -> None:
    stream_levelno = _level_number(level)
    export_levelno = min(stream_levelno, _level_number(export_level))

    common_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _AddService(service),
        structlog.processors.format_exc_info,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        # ExtraAdder keeps stdlib `extra=` fields, so they reach redaction instead of bypassing it.
        foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *common_processors],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *final_processors,
            # Redaction runs last, so nothing a final processor adds escapes it.
            *([RedactPii()] if redact_pii else []),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    # stderr, so a CLI keeps stdout for its own output.
    handler = _JsonLinesHandler(stream or sys.stderr, stream_levelno, export, export_levelno)
    handler.setFormatter(formatter)
    # On the handler, not a logger, so filters see the records of every logger.
    for log_filter in filters:
        handler.addFilter(log_filter)

    _install_handler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *common_processors,
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


def _level_number(name: str) -> int:
    level_name = name.upper()
    if level_name not in _LEVEL_NAMES:
        raise ValueError(
            f"unknown log level {level_name!r}; expected one of {sorted(_LEVEL_NAMES)}"
        )

    return logging.getLevelNamesMapping()[level_name]


def _install_handler(handler: logging.Handler) -> None:
    global _handler  # noqa: PLW0603 - one handler per process, replaced on reconfigure

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)

    root.addHandler(handler)
    root.setLevel(handler.level)
    _handler = handler
