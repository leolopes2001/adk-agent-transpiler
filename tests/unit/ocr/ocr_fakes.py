"""Fakes and builders shared by the OCR server unit tests."""

import base64
import io

import numpy as np
from PIL import Image

from ocr_mcp.engine import RecognizedLine
from ocr_mcp.settings import OcrSettings

TOKEN = "test-token-" + "x" * 32
SECRET = "test-secret-" + "y" * 32


class FakeOcrEngine:
    """Returns fixed lines whatever the image; records how it was called."""

    name = "fake"
    lang = "por"
    psm = 6

    def __init__(self, lines: list[RecognizedLine] | None = None) -> None:
        self.lines = lines or []
        self.calls: list[tuple[int, ...]] = []

    def version(self) -> str:
        return "0.0-test"

    def recognize(self, image: np.ndarray, timeout_s: float) -> list[RecognizedLine]:
        self.calls.append(image.shape)
        return list(self.lines)


def make_settings(**overrides: object) -> OcrSettings:
    values: dict[str, object] = {
        "mcp_auth_token": TOKEN,
        "pii_hmac_secret": SECRET,
        "host": "127.0.0.1",
        **overrides,
    }
    return OcrSettings(**values)


def png_base64(size: tuple[int, int] = (40, 30)) -> str:
    buffer = io.BytesIO()
    Image.new("L", size, 255).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
