"""Appointment rules.

The service knows nothing about HTTP or SQL: it talks to storage through the
`UnitOfWork` protocol below, which the infra layer implements with SQLAlchemy
and the unit tests implement with in-memory fakes.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from types import TracebackType
from typing import Protocol, Self

from scheduler_api.domain.catalog import Catalog
from scheduler_api.domain.models import (
    Appointment,
    AppointmentNotFoundError,
    AppointmentRequest,
    CatalogIssue,
    CatalogValidationError,
    ExamRef,
    new_appointment_id,
)
from scheduler_api.log import get_logger

_log = get_logger(__name__)


class AppointmentRepository(Protocol):
    def add(self, appointment: Appointment) -> None: ...

    def get(self, appointment_id: str) -> Appointment | None: ...


class UnitOfWork(Protocol):
    """One database transaction; leaving the block without `commit` rolls it back."""

    @property
    def appointments(self) -> AppointmentRepository: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...


def utc_now() -> datetime:
    return datetime.now(UTC)


class AppointmentService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        catalog: Catalog,
        *,
        clock: Callable[[], datetime] = utc_now,
        new_id: Callable[[], str] = new_appointment_id,
    ) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog
        self._clock = clock
        self._new_id = new_id

    def create(self, request: AppointmentRequest) -> Appointment:
        exams = self._validated_exams(request)
        appointment = Appointment(
            id=self._new_id(),
            status="requested",
            patient_ref=request.patient_ref,
            exams=exams,
            source_document_sha256=request.source_document_sha256,
            requested_by=request.requested_by,
            created_at=self._clock(),
        )
        with self._uow_factory() as uow:
            uow.appointments.add(appointment)
            uow.commit()
        _log.info("appointment_created", appointment_id=appointment.id, exam_count=len(exams))
        return appointment

    def get(self, appointment_id: str) -> Appointment:
        with self._uow_factory() as uow:
            appointment = uow.appointments.get(appointment_id)
        if appointment is None:
            raise AppointmentNotFoundError(appointment_id)
        return appointment

    def _validated_exams(self, request: AppointmentRequest) -> tuple[ExamRef, ...]:
        """Checks every exam against the catalog; returns them sorted, with official names."""
        issues: list[CatalogIssue] = []
        exams: list[ExamRef] = []
        for index, exam in enumerate(request.exams):
            entry = self._catalog.get(exam.code)
            if entry is None:
                issues.append(
                    CatalogIssue(index, "code", "unknown_exam_code", f"{exam.code} not found")
                )
            elif not self._catalog.name_matches(exam.code, exam.name):
                message = f"name does not match the catalog name for {exam.code}: {entry.name!r}"
                issues.append(CatalogIssue(index, "name", "exam_name_mismatch", message))
            else:
                exams.append(ExamRef(code=entry.code, name=entry.name))
        if issues:
            raise CatalogValidationError(issues)
        return tuple(sorted(exams, key=lambda exam: exam.code))
