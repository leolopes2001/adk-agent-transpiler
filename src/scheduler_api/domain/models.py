"""Entities and domain errors (plain dataclasses, no framework types)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ulid import ULID

AppointmentStatus = Literal["requested"]


def new_appointment_id() -> str:
    # ULID: sortable by creation time, but not guessable like a sequence.
    return f"apt_{ULID()}"


@dataclass(frozen=True, slots=True)
class ExamRef:
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class AppointmentRequest:
    patient_ref: str
    exams: tuple[ExamRef, ...]
    source_document_sha256: str
    requested_by: str


@dataclass(frozen=True, slots=True)
class Appointment:
    id: str
    status: AppointmentStatus
    patient_ref: str
    exams: tuple[ExamRef, ...]
    """Sorted by code, with the catalog's official names."""
    source_document_sha256: str
    requested_by: str
    created_at: datetime


class DomainError(Exception):
    """Base for errors the HTTP layer maps to a specific problem."""


IssueKind = Literal["unknown_exam_code", "exam_name_mismatch"]


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    index: int
    """Position of the exam in the request."""
    field: Literal["code", "name"]
    kind: IssueKind
    message: str


class CatalogValidationError(DomainError):
    def __init__(self, issues: list[CatalogIssue]) -> None:
        super().__init__(f"{len(issues)} exam(s) failed catalog validation")
        self.issues = issues

    def count(self, kind: IssueKind) -> int:
        return sum(1 for issue in self.issues if issue.kind == kind)


class AppointmentNotFoundError(DomainError):
    def __init__(self, appointment_id: str) -> None:
        super().__init__(appointment_id)
        self.appointment_id = appointment_id
