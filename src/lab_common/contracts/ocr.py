"""Contract of the OCR server's `extract_text` tool (version "1").

Every text field is already masked: the server applies the PII redactor before
building any of these objects, so they are safe to log, store in session
events or hand to a model.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

CONTRACT_VERSION: Literal["1"] = "1"
PATIENT_REF_PATTERN = r"^pt_[A-Z2-7]{26}$"
"""Same pattern the scheduler API accepts: "pt_" + 26 base32 characters."""

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


class OcrResult(_Contract):
    contract_version: Literal["1"] = CONTRACT_VERSION
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    text: str
    lines: list[OcrLine]
    mean_confidence: Score
    pii_counts: dict[str, NonNegativeInt]
    """Masked spans per entity name (`PERSON`, `CPF`, ...)."""
    patient_ref: Annotated[str, Field(pattern=PATIENT_REF_PATTERN)] | None
    """Pseudonymous patient id; None when the order has no patient identifier."""
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
