"""Logging setup of the OCR server.

On top of the shared JSON logging (with PII redaction), records from the
`mcp` and `uvicorn` libraries are scrubbed: their error messages can quote the
request that caused them. A malformed JSON-RPC call, for instance, is logged
by `mcp` as a pydantic error whose `input_value` holds part of the body,
`image_base64` included, and `mcp.shared.session` logs the whole rejected
message through the root logger. Those records keep their event text up to the
first ": " and the exception type only.

A record counts as coming from a library when its logger is named after it or
when it was emitted from a file inside the library (which catches calls on the
root logger).
"""

import logging
import os
from typing import TextIO

import mcp
import uvicorn

from lab_common import logging as lab_logging

SERVICE = "ocr-mcp"
REDACTED = "[REDACTED:PAYLOAD]"
LIBRARY_LOGGERS = ("mcp", "uvicorn")
_LIBRARY_DIRS = tuple(
    os.path.dirname(os.path.abspath(module.__file__)) + os.sep
    for module in (mcp, uvicorn)
    if module.__file__
)


def _from_library(record: logging.LogRecord) -> bool:
    if any(record.name == lib or record.name.startswith(f"{lib}.") for lib in LIBRARY_LOGGERS):
        return True
    return os.path.abspath(record.pathname).startswith(_LIBRARY_DIRS)


class LibraryLogFilter(logging.Filter):
    """Drops request payloads that `mcp` and `uvicorn` put into their log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not _from_library(record):
            return True
        # uvicorn duplicates its message with ANSI colors for its own formatter.
        record.__dict__.pop("color_message", None)
        # Set by any handler that formatted the record before ours; structlog
        # would pick it up as an extra field holding the unredacted message.
        record.__dict__.pop("message", None)
        if record.exc_info:
            exc_type = record.exc_info[0]
            record.exception_type = exc_type.__name__ if exc_type else None
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            record.exception = REDACTED
        # Messages are often f-strings that embed an exception or the request
        # after a colon ("Received exception from stream: <pydantic error with
        # the body>"), at any level, including debug dumps of the message.
        message = record.getMessage()
        head, separator, _ = message.partition(": ")
        record.msg = f"{head}{separator}{REDACTED}" if separator else message
        record.args = None
        return True


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    lab_logging.configure(SERVICE, level, stream=stream, filters=[LibraryLogFilter()])
