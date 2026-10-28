"""Run the OCR MCP server: `python -m ocr_mcp` (settings from the environment)."""

import uvicorn

from lab_common import telemetry
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.logs import SERVICE, configure_logging
from ocr_mcp.server import build_app
from ocr_mcp.settings import OcrSettings


def main() -> None:
    settings = OcrSettings()  # type: ignore[call-arg]  # fields come from the environment
    exporting = telemetry.setup(SERVICE)
    configure_logging(settings.log_level, export=exporting.log_export if exporting else None)
    app = build_app(settings, TesseractEngine())
    # log_config=None keeps uvicorn's loggers on the root JSON handler (with PII
    # redaction); the access log is off because paths carry session ids only.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
