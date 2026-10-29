from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_core import PydanticCustomError

from scheduler_api.domain import models

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
        description="Código do catálogo.",
        examples=["LAB-10001"],
    ),
]


class _Contract(BaseModel):
    # Unknown fields are rejected, not dropped: the contract has no place for personal data.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Exam(_Contract):
    """Exame pedido: código e nome do catálogo."""

    code: ExamCode
    name: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Nome do exame no catálogo; comparado sem diferenciar maiúsculas, acentos, "
            "pontuação e espaços."
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


class NewAppointment(_Contract):
    """Solicitação de agendamento de um paciente a partir de um documento."""

    patient_ref: str = Field(
        pattern=PATIENT_REF_PATTERN,
        description="Pseudônimo opaco do paciente (nunca nome ou documento).",
        examples=["pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"],
    )
    exams: Annotated[list[Exam], AfterValidator(_reject_duplicate_codes)] = Field(
        min_length=1,
        max_length=MAX_EXAMS,
        description=f"De 1 a {MAX_EXAMS} exames, cada código no máximo uma vez.",
    )
    source_document_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 do documento de origem, em hexadecimal minúsculo.",
        examples=["9b1e" + "0" * 60],
    )
    requested_by: str = Field(
        pattern=REQUESTER_PATTERN,
        description="Identificador do sistema solicitante.",
        examples=["exam_scheduling_agent"],
    )

    def to_domain(self) -> models.AppointmentRequest:
        return models.AppointmentRequest(
            patient_ref=self.patient_ref,
            exams=tuple(models.ExamRef(code=exam.code, name=exam.name) for exam in self.exams),
            source_document_sha256=self.source_document_sha256,
            requested_by=self.requested_by,
        )


class Appointment(_Contract):
    """Solicitação de agendamento registrada."""

    appointment_id: str = Field(
        pattern=APPOINTMENT_ID_PATTERN,
        description="Id da solicitação.",
        examples=["apt_" + "0" * 26],
    )
    status: Literal["requested"] = Field(
        description="A solicitação é registrada, ainda não agendada."
    )
    patient_ref: str = Field(description="Pseudônimo do paciente.")
    exams: list[Exam] = Field(description="Ordenados por código, com o nome oficial do catálogo.")
    source_document_sha256: str = Field(description="SHA-256 do documento de origem.")
    requested_by: str = Field(description="Sistema solicitante.")
    created_at: datetime = Field(description="Data e hora de criação, em UTC.")

    @classmethod
    def from_domain(cls, appointment: models.Appointment) -> "Appointment":
        return cls(
            appointment_id=appointment.id,
            status=appointment.status,
            patient_ref=appointment.patient_ref,
            exams=[Exam(code=exam.code, name=exam.name) for exam in appointment.exams],
            source_document_sha256=appointment.source_document_sha256,
            requested_by=appointment.requested_by,
            created_at=appointment.created_at,
        )


class CatalogEntry(_Contract):
    """Exame do catálogo."""

    code: str = Field(description="Código do catálogo.")
    name: str = Field(description="Nome oficial.")
