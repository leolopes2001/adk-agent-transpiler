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
from ocr_mcp.settings import DEFAULT_MAX_PIXELS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

# Pillow's guard is only a backstop; the explicit pixel check applies the configured limit.
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_PIXELS


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    pixels: np.ndarray
    sha256: str
    """Of the decoded file bytes; identifies the document without its content."""
    format: str
    size_bytes: int


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


def decode_base64(data: str, max_bytes: int) -> bytes:
    # Rejecting on the encoded length avoids allocating an oversized buffer.
    if len(data) > _max_base64_length(max_bytes):
        raise _too_large(max_bytes)

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise OcrError(
            ErrorCode.INVALID_BASE64,
            "image_base64 is not valid base64",
        ) from None

    if not raw:
        raise OcrError(
            ErrorCode.INVALID_BASE64,
            "image_base64 is empty",
        )

    if len(raw) > max_bytes:
        raise _too_large(max_bytes)

    return raw


def detect_format(raw: bytes) -> str:
    if raw.startswith(PNG_MAGIC):
        return "PNG"

    if raw.startswith(JPEG_MAGIC):
        return "JPEG"

    raise OcrError(
        ErrorCode.UNSUPPORTED_FORMAT,
        "only PNG and JPEG images are accepted",
    )


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
            return np.asarray(_grayscale_on_white(upright), dtype=np.uint8)
    except OcrError:
        raise
    except Image.DecompressionBombError:
        raise _too_many_pixels(max_pixels) from None
    except Exception:  # noqa: BLE001 - any decoder failure means the file is unusable
        raise _corrupted() from None


def _max_base64_length(max_bytes: int) -> int:
    three_byte_groups = -(-max_bytes // 3)
    return 4 * three_byte_groups


def _too_large(max_bytes: int) -> OcrError:
    return OcrError(
        ErrorCode.IMAGE_TOO_LARGE,
        f"image exceeds {max_bytes} bytes",
    )


def _too_many_pixels(max_pixels: int) -> OcrError:
    return OcrError(
        ErrorCode.IMAGE_TOO_MANY_PIXELS,
        f"image exceeds {max_pixels} pixels",
    )


def _corrupted() -> OcrError:
    return OcrError(
        ErrorCode.CORRUPTED_IMAGE,
        "image could not be decoded",
    )


def _grayscale_on_white(image: Image.Image) -> Image.Image:
    """A white background keeps transparent areas from turning black."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    return image.convert("L")
