import time

import uvicorn

from lab_common import logging as lab_logging
from lab_common import telemetry
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.logs import SERVICE, configure_logging
from ocr_mcp.server import build_app
from ocr_mcp.settings import OcrSettings
from pii_guard import PiiRedactor, document_detectors

log = lab_logging.get_logger(__name__)


def main() -> None:
    settings = OcrSettings()  # type: ignore[call-arg]
    exporting = telemetry.setup(SERVICE)
    configure_logging(settings.log_level, export=exporting.log_export if exporting else None)

    # Before the server starts, so /health answers only once the NER model is loaded;
    # a model that fails to load stops the process instead of serving unmasked names.
    started = time.perf_counter()
    redactor = PiiRedactor(document_detectors())
    log.info("pii_model_loaded", duration_ms=round((time.perf_counter() - started) * 1000))

    app = build_app(settings, TesseractEngine(), redactor)

    # Keep uvicorn on the redacting JSON handler; no access log since paths carry session ids.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
