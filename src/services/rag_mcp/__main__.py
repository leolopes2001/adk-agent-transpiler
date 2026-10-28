import sys

import uvicorn

from lab_common import logging as lab_logging
from lab_common import telemetry
from rag_mcp.catalog import CatalogError, load_catalog
from rag_mcp.logs import SERVICE, configure_logging
from rag_mcp.server import build_app
from rag_mcp.settings import RagSettings

log = lab_logging.get_logger(__name__)


def main() -> None:
    settings = RagSettings()  # type: ignore[call-arg]  # fields come from the environment
    exporting = telemetry.setup(SERVICE)
    configure_logging(settings.log_level, export=exporting.log_export if exporting else None)

    try:
        catalog = load_catalog(settings.catalog_path)
    except CatalogError as error:
        log.error("catalog_invalid", reason=str(error))
        sys.exit(1)

    log.info(
        "catalog_loaded",
        catalog_version=catalog.version,
        catalog_sha256=catalog.sha256,
        exam_count=len(catalog),
    )
    app = build_app(settings, catalog)
    # Keeps uvicorn on the PII-redacting JSON handler; access log paths only carry session ids.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
