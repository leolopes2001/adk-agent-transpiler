"""Errors the `extract_text` tool reports, as `{code, message, retryable}`.

Messages are fixed strings chosen here, never built from the input or from a
lower-level exception, so an error can be logged and returned without risk of
carrying document content.
"""

from lab_common.contracts.ocr import RETRYABLE_ERRORS, OcrToolError
from lab_common.contracts.ocr import OcrErrorCode as ErrorCode

__all__ = ["ErrorCode", "OcrError"]


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
