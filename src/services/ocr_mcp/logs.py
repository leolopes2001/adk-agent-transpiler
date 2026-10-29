from typing import TextIO

from lab_common import logging as lab_logging
from lab_common.mcp_server import LIBRARY_LOGGERS, REDACTED, LibraryLogFilter

__all__ = ["REDACTED", "SERVICE", "LibraryLogFilter", "configure_logging"]

SERVICE = "ocr-mcp"
# Presidio and spaCy read the document; their records are redacted like the server's.
OCR_LIBRARY_LOGGERS = (*LIBRARY_LOGGERS, "presidio-analyzer", "presidio_analyzer", "spacy")


def configure_logging(
    level: str = "INFO",
    stream: TextIO | None = None,
    export: lab_logging.LogExport | None = None,
) -> None:
    lab_logging.configure(
        SERVICE,
        level,
        stream=stream,
        filters=[LibraryLogFilter(OCR_LIBRARY_LOGGERS)],
        export=export,
    )
