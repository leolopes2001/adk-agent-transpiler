"""Run the RAG MCP server: `python -m rag_mcp` (settings from the environment).

The catalog is loaded and validated before the server binds its port; an
invalid catalog is logged and the process exits with status 1.
"""

import sys

import uvicorn

from lab_common import logging as lab_logging
from rag_mcp.catalog import CatalogError, load_catalog
from rag_mcp.logs import configure_logging
from rag_mcp.server import build_app
from rag_mcp.settings import RagSettings

log = lab_logging.get_logger(__name__)


def main() -> None:
    settings = RagSettings()  # type: ignore[call-arg]  # fields come from the environment
    configure_logging(settings.log_level)
    try:
        catalog = load_catalog(settings.catalog_path)
    except CatalogError as error:
        log.error("catalog_invalid", reason=str(error))
        sys.exit(1)
    # The hash is left out: the log redaction can mistake its digit runs for a
    # phone number; /health and catalog://info report it intact.
    log.info("catalog_loaded", catalog_version=catalog.version, exam_count=len(catalog))
    app = build_app(settings, catalog)
    # log_config=None keeps uvicorn's loggers on the root JSON handler (with PII
    # redaction); the access log is off because paths carry session ids only.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
