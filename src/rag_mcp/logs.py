"""Logging setup of the RAG server.

Shared JSON logging (with PII redaction) plus `LibraryLogFilter`, which keeps
request payloads that the `mcp` and `uvicorn` libraries put into their error
messages out of the log (see `lab_common.mcp_server`).
"""

from typing import TextIO

from lab_common import logging as lab_logging
from lab_common.mcp_server import REDACTED, LibraryLogFilter

__all__ = ["REDACTED", "SERVICE", "LibraryLogFilter", "configure_logging"]

SERVICE = "rag-mcp"


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    lab_logging.configure(SERVICE, level, stream=stream, filters=[LibraryLogFilter()])
