from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

CONTRACT_VERSION: Literal["1"] = "1"
# Must match the patient reference the scheduler API accepts.
PATIENT_REF_PATTERN = r"^pt_[A-Z2-7]{26}$"

Score = Annotated[float, Field(ge=0, le=1)]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OcrLine(_Contract):
    text: str
    confidence: Score


class EngineInfo(_Contract):
    name: str
    lang: str
    psm: int


# Every text field is already masked: the server redacts PII before building it.
class OcrResult(_Contract):
    contract_version: Literal["1"] = CONTRACT_VERSION
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    text: str
    lines: list[OcrLine]
    mean_confidence: Score
    pii_counts: dict[str, NonNegativeInt]
    patient_ref: Annotated[str, Field(pattern=PATIENT_REF_PATTERN)] | None
    engine: EngineInfo
    timings_ms: dict[str, NonNegativeInt]


class OcrErrorCode(StrEnum):
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


RETRYABLE_ERRORS = frozenset({OcrErrorCode.OCR_TIMEOUT, OcrErrorCode.BUSY})


class OcrToolError(_Contract):
    """Body of an `extract_text` error result (`isError: true`); never has text."""

    code: OcrErrorCode
    message: str
    retryable: bool
