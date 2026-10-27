"""Run the OCR MCP server: `python -m ocr_mcp` (settings from the environment)."""

import uvicorn

from lab_common import logging as lab_logging
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.server import SERVICE, build_app
from ocr_mcp.settings import OcrSettings


def main() -> None:
    settings = OcrSettings()  # type: ignore[call-arg]  # fields come from the environment
    lab_logging.configure(SERVICE, settings.log_level)
    app = build_app(settings, TesseractEngine())
    # log_config=None keeps uvicorn's loggers on the root JSON handler (with PII
    # redaction); the access log is off because paths carry session ids only.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
