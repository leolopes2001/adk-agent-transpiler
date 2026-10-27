import base64
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_mcp.errors import ErrorCode, OcrError
from ocr_mcp.validation import ValidatedImage, decode_base64, detect_format, validate_image

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES = REPO_ROOT / "tests" / "data" / "images"
SAMPLE = REPO_ROOT / "samples" / "pedido_medico_ficticio.png"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def validate(
    data: bytes | str, max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS
) -> ValidatedImage:
    encoded = data if isinstance(data, str) else b64(data)
    return validate_image(encoded, max_bytes=max_bytes, max_pixels=max_pixels)


def error_code(data: bytes | str, **limits: int) -> ErrorCode:
    with pytest.raises(OcrError) as excinfo:
        validate(data, **limits)
    return excinfo.value.code


def encode(image: Image.Image, image_format: str, **params: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **params)
    return buffer.getvalue()


class TestBase64:
    @pytest.mark.parametrize("data", ["not base64!", "iVBORw0KGgo", "ção", "aGVsbG8=\n", ""])
    def test_rejects_invalid_base64(self, data: str) -> None:
        assert error_code(data) is ErrorCode.INVALID_BASE64

    def test_rejects_oversized_input_before_decoding(self) -> None:
        with pytest.raises(OcrError) as excinfo:
            decode_base64("A" * 20, max_bytes=12)
        assert excinfo.value.code is ErrorCode.IMAGE_TOO_LARGE

    def test_rejects_decoded_size_over_limit(self) -> None:
        # 16 base64 characters pass the length pre-check for 10 bytes but decode to 12.
        assert error_code(b64(b"x" * 12), max_bytes=10) is ErrorCode.IMAGE_TOO_LARGE

    def test_accepts_exact_limit(self) -> None:
        assert decode_base64(b64(b"x" * 12), max_bytes=12) == b"x" * 12


class TestFormat:
    @pytest.mark.parametrize(
        "data", [b"GIF89a" + b"\0" * 20, b"%PDF-1.7", b"BM" + b"\0" * 20, b"\x89PNX\r\n\x1a\n"]
    )
    def test_rejects_other_formats(self, data: bytes) -> None:
        assert error_code(data) is ErrorCode.UNSUPPORTED_FORMAT

    def test_detects_png_and_jpeg(self) -> None:
        assert detect_format(SAMPLE.read_bytes()) == "PNG"
        assert detect_format((IMAGES / "valid.jpg").read_bytes()) == "JPEG"


class TestPixels:
    def test_decompression_bomb_is_rejected_quickly(self) -> None:
        started = time.perf_counter()
        assert error_code((IMAGES / "decompression_bomb.png").read_bytes()) is (
            ErrorCode.IMAGE_TOO_MANY_PIXELS
        )
        assert time.perf_counter() - started < 1.0

    def test_configured_limit_applies(self) -> None:
        image = encode(Image.new("L", (100, 100), 255), "PNG")
        assert error_code(image, max_pixels=9_999) is ErrorCode.IMAGE_TOO_MANY_PIXELS
        assert validate(image, max_pixels=10_000).pixels.shape == (100, 100)


class TestCorrupted:
    def test_truncated_png(self) -> None:
        assert error_code((IMAGES / "corrupted.png").read_bytes()) is ErrorCode.CORRUPTED_IMAGE

    def test_jpeg_magic_with_garbage(self) -> None:
        assert error_code(b"\xff\xd8\xff" + b"\x00garbage" * 10) is ErrorCode.CORRUPTED_IMAGE

    def test_png_signature_on_jpeg_content(self) -> None:
        jpeg = (IMAGES / "valid.jpg").read_bytes()
        assert error_code(b"\x89PNG\r\n\x1a\n" + jpeg[3:]) is ErrorCode.CORRUPTED_IMAGE


class TestValidImages:
    def test_sample_png(self) -> None:
        raw = SAMPLE.read_bytes()
        result = validate(raw)

        assert result.pixels.shape == (2339, 1654)
        assert result.pixels.dtype == np.uint8
        assert result.format == "PNG"
        assert result.size_bytes == len(raw)
        assert result.sha256 == hashlib.sha256(raw).hexdigest()

    def test_jpeg(self) -> None:
        result = validate((IMAGES / "valid.jpg").read_bytes())
        assert result.format == "JPEG"
        assert result.pixels.ndim == 2

    def test_color_image_becomes_grayscale(self) -> None:
        result = validate(encode(Image.new("RGB", (10, 10), (255, 0, 0)), "PNG"))
        assert result.pixels.shape == (10, 10)

    def test_transparent_background_becomes_white(self) -> None:
        result = validate(encode(Image.new("RGBA", (10, 10), (0, 0, 0, 0)), "PNG"))
        assert (result.pixels == 255).all()

    def test_exif_orientation_is_applied(self) -> None:
        exif = Image.Exif()
        exif[0x0112] = 6  # rotated 90 degrees clockwise
        image = encode(Image.new("L", (40, 20), 255), "JPEG", exif=exif.tobytes())
        assert validate(image).pixels.shape == (40, 20)


def test_error_messages_never_echo_the_input() -> None:
    payloads = ["Paciente Maria!", b64(b"Paciente Maria"), b64(b"\xff\xd8\xffPaciente Maria")]
    for payload in payloads:
        with pytest.raises(OcrError) as excinfo:
            validate(payload)
        assert "Maria" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.as_dict()["retryable"] is False
