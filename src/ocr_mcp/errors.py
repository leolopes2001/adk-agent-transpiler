"""Errors the `extract_text` tool reports, as `{code, message, retryable}`.

Messages are fixed strings chosen here, never built from the input or from a
lower-level exception, so an error can be logged and returned without risk of
carrying document content.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_BASE64 = "INVALID_BASE64"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    IMAGE_TOO_MANY_PIXELS = "IMAGE_TOO_MANY_PIXELS"
    CORRUPTED_IMAGE = "CORRUPTED_IMAGE"
    PII_ENGINE_FAILURE = "PII_ENGINE_FAILURE"
    OCR_TIMEOUT = "OCR_TIMEOUT"
    BUSY = "BUSY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


RETRYABLE = frozenset({ErrorCode.OCR_TIMEOUT, ErrorCode.BUSY})


class OcrError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def as_dict(self) -> dict[str, str | bool]:
        return {"code": self.code.value, "message": self.message, "retryable": self.retryable}
