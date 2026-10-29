import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from ulid import ULID

AppointmentStatus = Literal["requested"]
IssueKind = Literal["unknown_exam_code", "exam_name_mismatch"]


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
    source_document_sha256: str
    requested_by: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    request_hash: str
    appointment_id: str
    response_status: int
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    index: int
    field: Literal["code", "name"]
    kind: IssueKind
    message: str


class DomainError(Exception):
    pass


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


class IdempotencyKeyReuseError(DomainError):
    pass


class IdempotencyKeyTakenError(Exception):
    """Not a `DomainError`: the service retries once, so it only escapes as a 500."""


def new_appointment_id() -> str:
    # ULID: sortable by creation time, but not guessable like a sequence.
    return f"apt_{ULID()}"


def canonical_request_hash(request: AppointmentRequest) -> str:
    canonical = asdict(request)
    canonical["exams"] = sorted(canonical["exams"], key=lambda exam: exam["code"])
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
