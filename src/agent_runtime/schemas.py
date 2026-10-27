"""Data contracts between workflow nodes.

Node outputs become ADK session events, so none of these types may carry raw
PII or image bytes: OCR text arrives already masked and the image never
leaves the node that reads it.

Every port type name in the transpiler registry has a counterpart here.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[float, Field(ge=0, le=1)]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunInput(_Schema):
    image_path: str


class OcrLine(_Schema):
    text: str
    confidence: Score


class OcrDocument(_Schema):
    document_sha256: Sha256
    text: str
    """Masked by the OCR server."""
    lines: list[OcrLine]
    mean_confidence: Score
    pii_counts: dict[str, int]
    patient_ref: str | None


class InjectionReport(_Schema):
    severity: Literal["none", "low", "medium", "high"]
    rule_ids: list[str]


class GuardedDocument(_Schema):
    document_sha256: Sha256
    text: str
    lines: list[OcrLine]
    injection: InjectionReport


class ExtractedExam(_Schema):
    name: Annotated[str, Field(min_length=2, max_length=120)]
    evidence: Annotated[str, Field(min_length=1, max_length=200)]
    """Text copied from the document; checked for grounding by reconcile."""


class ExtractedExams(_Schema):
    exams: Annotated[list[ExtractedExam], Field(max_length=30)]


LlmMentions = ExtractedExams
"""Port name for the LLM extractor output; same schema the Agent is constrained to."""


class ExamMention(_Schema):
    text: str
    source: Literal["llm", "lexical", "both"]
    evidence: str


class ExamMentions(_Schema):
    items: list[ExamMention]


type JoinResult = dict[str, Any]
"""What a JoinNode emits: each predecessor's output keyed by node name."""


class Candidate(_Schema):
    code: Annotated[str, Field(pattern=r"^LAB-\d{5}$")]
    name: str
    score: Score
    matched_on: str


class ResolvedExam(_Schema):
    mention: ExamMention
    candidates: list[Candidate]


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
