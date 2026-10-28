from typing import TextIO

from lab_common import logging as lab_logging
from lab_common.mcp_server import REDACTED, LibraryLogFilter

__all__ = ["REDACTED", "SERVICE", "LibraryLogFilter", "configure_logging"]

SERVICE = "ocr-mcp"


def configure_logging(
    level: str = "INFO",
    stream: TextIO | None = None,
    export: lab_logging.LogExport | None = None,
) -> None:
    lab_logging.configure(
        SERVICE, level, stream=stream, filters=[LibraryLogFilter()], export=export
    )
