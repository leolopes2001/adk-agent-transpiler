from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_core import PydanticCustomError

from scheduler_api.domain.models import Appointment, AppointmentRequest, ExamRef

EXAM_CODE_PATTERN = r"^[A-Z]{3}-\d{5}$"
PATIENT_REF_PATTERN = r"^pt_[A-Z2-7]{26}$"
APPOINTMENT_ID_PATTERN = r"^apt_[0-9A-HJKMNP-TV-Z]{26}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
REQUESTER_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
MAX_EXAMS = 30
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100

ExamCode = Annotated[
    str,
    Field(
        pattern=EXAM_CODE_PATTERN,
        description="Catalog code.",
        examples=["LAB-10001"],
    ),
]


class _Contract(BaseModel):
    # Unknown fields are rejected, not dropped: the contract has no place for personal data.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Exam(_Contract):
    code: ExamCode
    name: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Exam name as in the catalog; compared ignoring case, accents, punctuation and spacing."
        ),
        examples=["Hemograma completo"],
    )


def _reject_duplicate_codes(exams: list[Exam]) -> list[Exam]:
    seen: set[str] = set()
    for exam in exams:
        if exam.code in seen:
            raise PydanticCustomError(
                "duplicate_exam_code",
                "exam code {code} appears more than once",
                {"code": exam.code},
            )
        seen.add(exam.code)

    return exams


class AppointmentCreate(_Contract):
    """Scheduling request for one patient and one source document."""

    patient_ref: str = Field(
        pattern=PATIENT_REF_PATTERN,
        description="Opaque patient pseudonym (never a name or document number).",
        examples=["pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"],
    )
    exams: Annotated[list[Exam], AfterValidator(_reject_duplicate_codes)] = Field(
        min_length=1,
        max_length=MAX_EXAMS,
        description=f"1 to {MAX_EXAMS} exams, each code at most once.",
    )
    source_document_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 of the source document, lowercase hex.",
        examples=["9b1e" + "0" * 60],
    )
    requested_by: str = Field(
        pattern=REQUESTER_PATTERN,
        description="Identifier of the requesting system.",
        examples=["exam_scheduling_agent"],
    )

    def to_domain(self) -> AppointmentRequest:
        return AppointmentRequest(
            patient_ref=self.patient_ref,
            exams=tuple(ExamRef(code=exam.code, name=exam.name) for exam in self.exams),
            source_document_sha256=self.source_document_sha256,
            requested_by=self.requested_by,
        )


class AppointmentOut(_Contract):
    appointment_id: str = Field(pattern=APPOINTMENT_ID_PATTERN, examples=["apt_" + "0" * 26])
    status: Literal["requested"] = Field(description="Requests are recorded, not yet booked.")
    patient_ref: str
    exams: list[Exam] = Field(description="Sorted by code, with the catalog's official names.")
    source_document_sha256: str
    requested_by: str
    created_at: datetime = Field(description="UTC.")

    @classmethod
    def from_domain(cls, appointment: Appointment) -> "AppointmentOut":
        return cls(
            appointment_id=appointment.id,
            status=appointment.status,
            patient_ref=appointment.patient_ref,
            exams=[Exam(code=exam.code, name=exam.name) for exam in appointment.exams],
            source_document_sha256=appointment.source_document_sha256,
            requested_by=appointment.requested_by,
            created_at=appointment.created_at,
        )


class ExamOut(_Contract):
    code: str
    name: str
