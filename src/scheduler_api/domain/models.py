"""Entities and domain errors (plain dataclasses, no framework types)."""

import hashlib
import json
from dataclasses import asdict, dataclass
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


def canonical_request_hash(request: AppointmentRequest) -> str:
    """SHA-256 of the request as canonical JSON: sorted keys, exams sorted by code.

    The order in which the client lists exams does not change the hash; any
    other difference in the body does.
    """
    canonical = asdict(request)
    canonical["exams"] = sorted(canonical["exams"], key=lambda exam: exam["code"])
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    request_hash: str
    appointment_id: str
    response_status: int
    created_at: datetime
    expires_at: datetime


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


class IdempotencyKeyReuseError(DomainError):
    """The key was already used, within its TTL, for a different request body."""


class IdempotencyKeyTakenError(Exception):
    """Raised by storage when a concurrent request committed the same key first.

    Not a `DomainError`: the service handles it by re-reading the key, and it
    only escapes (as a 500) if the same race happens twice in a row.
    """
