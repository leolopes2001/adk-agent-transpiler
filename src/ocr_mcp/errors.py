"""Messages are fixed strings, never built from input, so they cannot carry document content."""

from lab_common.contracts.ocr import RETRYABLE_ERRORS, OcrToolError
from lab_common.contracts.ocr import OcrErrorCode as ErrorCode

__all__ = ["OCR_TIMEOUT_MESSAGE", "ErrorCode", "OcrError"]

OCR_TIMEOUT_MESSAGE = "OCR exceeded the time limit"


class OcrError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_ERRORS

    def as_dict(self) -> dict[str, str | bool]:
        return OcrToolError(
            code=self.code, message=self.message, retryable=self.retryable
        ).model_dump(mode="json")
