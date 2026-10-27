"""Decoding and validation of the uploaded image, before any pixel is processed.

Checks run from cheapest to most expensive: encoded length, base64, magic
bytes, declared dimensions (read from the header only, so a decompression
bomb is rejected before it is decoded), `verify()`, and finally a full decode.
The server repeats all of these even though the client validates too.
"""

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from ocr_mcp.errors import ErrorCode, OcrError

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

# Pillow's own guard (warning above this, error above twice this) backs up the
# explicit check below; the explicit check is the one that applies the setting.
Image.MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    pixels: np.ndarray
    """Grayscale, uint8, upright (EXIF orientation applied)."""
    sha256: str
    """Of the decoded file bytes; identifies the document without its content."""
    format: str
    size_bytes: int


def decode_base64(data: str, max_bytes: int) -> bytes:
    # Rejecting on the encoded length avoids allocating an oversized buffer.
    if len(data) > 4 * (-(-max_bytes // 3)):
        raise OcrError(ErrorCode.IMAGE_TOO_LARGE, f"image exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise OcrError(ErrorCode.INVALID_BASE64, "image_base64 is not valid base64") from None
    if not raw:
        raise OcrError(ErrorCode.INVALID_BASE64, "image_base64 is empty")
    if len(raw) > max_bytes:
        raise OcrError(ErrorCode.IMAGE_TOO_LARGE, f"image exceeds {max_bytes} bytes")
    return raw


def detect_format(raw: bytes) -> str:
    if raw.startswith(PNG_MAGIC):
        return "PNG"
    if raw.startswith(JPEG_MAGIC):
        return "JPEG"
    raise OcrError(ErrorCode.UNSUPPORTED_FORMAT, "only PNG and JPEG images are accepted")


def _too_many_pixels(max_pixels: int) -> OcrError:
    return OcrError(ErrorCode.IMAGE_TOO_MANY_PIXELS, f"image exceeds {max_pixels} pixels")


def _corrupted() -> OcrError:
    return OcrError(ErrorCode.CORRUPTED_IMAGE, "image could not be decoded")


def _flatten(image: Image.Image) -> Image.Image:
    """Grayscale on a white background, so transparent areas do not turn black."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("L")


def load_pixels(raw: bytes, image_format: str, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw), formats=[image_format]) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise _too_many_pixels(max_pixels)
            image.verify()
        # verify() leaves the image unusable, so it is opened again to decode.
        with Image.open(io.BytesIO(raw), formats=[image_format]) as image:
            image.load()
            upright = ImageOps.exif_transpose(image)
            return np.asarray(_flatten(upright), dtype=np.uint8)
    except OcrError:
        raise
    except Image.DecompressionBombError:
        raise _too_many_pixels(max_pixels) from None
    except Exception:  # noqa: BLE001 - any decoder failure means the file is unusable
        raise _corrupted() from None


def validate_image(image_base64: str, *, max_bytes: int, max_pixels: int) -> ValidatedImage:
    raw = decode_base64(image_base64, max_bytes)
    image_format = detect_format(raw)
    pixels = load_pixels(raw, image_format, max_pixels)
    return ValidatedImage(
        pixels=pixels,
        sha256=hashlib.sha256(raw).hexdigest(),
        format=image_format,
        size_bytes=len(raw),
    )
