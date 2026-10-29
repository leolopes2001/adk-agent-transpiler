from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lab_common.contracts import rag
from lab_common.contracts.ocr import OcrLine, OcrResult

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


# Node outputs become ADK session events: no raw PII or image bytes in any of these types.
class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunInput(_Schema):
    image_path: str


# Port names of the transpiler registry; a contract test keeps both sides in sync.
OcrDocument = OcrResult


class InjectionReport(_Schema):
    severity: Literal["none", "low", "medium", "high"]
    rule_ids: list[str]


class GuardedDocument(_Schema):
    document_sha256: Sha256
    text: str
    lines: list[OcrLine]
    injection: InjectionReport


def _without_additional_properties(schema: dict[str, Any]) -> None:
    schema.pop("additionalProperties", None)


class _LlmSchema(_Schema):
    """Schema the model must answer in (`Agent.output_schema`).

    Validation still forbids unknown fields; only the JSON schema sent to the
    model leaves out `additionalProperties`, which the Gemini API rejects in a
    response schema (400 INVALID_ARGUMENT).
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_extra=_without_additional_properties
    )


class ExtractedExam(_LlmSchema):
    name: Annotated[str, Field(min_length=2, max_length=120)]
    evidence: Annotated[str, Field(min_length=1, max_length=200)]


class ExtractedExams(_LlmSchema):
    exams: Annotated[list[ExtractedExam], Field(max_length=30)]


LlmMentions = ExtractedExams


class ExamMention(_Schema):
    text: str
    source: Literal["llm", "lexical", "both"]
    evidence: str


class ExamMentions(_Schema):
    items: list[ExamMention]


type JoinResult = dict[str, Any]

# Catalog codes only ever come from the RAG server, never from the model.
Candidate = rag.Candidate


class ResolvedExam(_Schema):
    mention: ExamMention
    candidates: list[Candidate]
    evidence_candidates: list[Candidate] | None = None


class ResolvedExams(_Schema):
    items: list[ResolvedExam]


class GateItem(_Schema):
    mention: ExamMention
    status: Literal["accepted", "review", "rejected"]
    selected: Candidate | None
    candidates: list[Candidate]
    reason: str | None


class GateDecision(_Schema):
    items: list[GateItem]


class ScheduleOutcome(_Schema):
    decision: GateDecision
    status: Literal["scheduled", "replayed", "not_scheduled", "failed"]
    appointment_id: str | None
    reason: str | None


class RunFailure(_Schema):
    code: Literal["OCR_LOW_QUALITY", "INPUT_BLOCKED", "NO_EXAMS_RESOLVED"]
    user_message: str
