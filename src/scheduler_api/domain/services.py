"""Appointment rules.

The service knows nothing about HTTP or SQL: it talks to storage through the
`UnitOfWork` protocol below, which the infra layer implements with SQLAlchemy
and the unit tests implement with in-memory fakes.

Idempotency (ADR-0006): each create runs in one transaction that reads the
key, and either replays the stored result, rejects a different body, or
inserts the appointment and the key together. The key is the primary key of
its table, so two concurrent requests cannot both insert it: the loser gets
`IdempotencyKeyTakenError`, rolls back and runs the read once more, which now
finds the winner's record.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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
    IdempotencyKeyReuseError,
    IdempotencyKeyTakenError,
    IdempotencyRecord,
    canonical_request_hash,
    new_appointment_id,
)
from scheduler_api.log import get_logger

_log = get_logger(__name__)


class AppointmentRepository(Protocol):
    def add(self, appointment: Appointment) -> None: ...

    def get(self, appointment_id: str) -> Appointment | None: ...


class IdempotencyRepository(Protocol):
    def get(self, key: str) -> IdempotencyRecord | None: ...

    def add(self, record: IdempotencyRecord) -> None:
        """Raises `IdempotencyKeyTakenError` if the key already exists."""

    def delete_expired(self, key: str, now: datetime) -> None: ...


class UnitOfWork(Protocol):
    """One database transaction; leaving the block without `commit` rolls it back."""

    @property
    def appointments(self) -> AppointmentRepository: ...

    @property
    def idempotency(self) -> IdempotencyRepository: ...

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
        idempotency_ttl: timedelta = timedelta(hours=24),
        clock: Callable[[], datetime] = utc_now,
        new_id: Callable[[], str] = new_appointment_id,
    ) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog
        self._ttl = idempotency_ttl
        self._clock = clock
        self._new_id = new_id

    def create(self, request: AppointmentRequest, idempotency_key: str) -> tuple[Appointment, bool]:
        """Returns the appointment and whether it is a replay of an earlier request."""
        # Invalid requests never reach storage, whatever their key.
        exams = self._validated_exams(request)
        request_hash = canonical_request_hash(request)
        try:
            return self._create_once(request, exams, idempotency_key, request_hash)
        except IdempotencyKeyTakenError:
            _log.info("idempotency_race_retry")
            return self._create_once(request, exams, idempotency_key, request_hash)

    def _create_once(
        self,
        request: AppointmentRequest,
        exams: tuple[ExamRef, ...],
        key: str,
        request_hash: str,
    ) -> tuple[Appointment, bool]:
        with self._uow_factory() as uow:
            now = self._clock()
            record = uow.idempotency.get(key)
            if record is not None and record.expires_at > now:
                return self._replay(uow, record, request_hash), True
            if record is not None:
                # Expired: the key starts over and a new appointment is created.
                uow.idempotency.delete_expired(key, now)

            appointment = Appointment(
                id=self._new_id(),
                status="requested",
                patient_ref=request.patient_ref,
                exams=exams,
                source_document_sha256=request.source_document_sha256,
                requested_by=request.requested_by,
                created_at=now,
            )
            uow.appointments.add(appointment)
            uow.idempotency.add(
                IdempotencyRecord(
                    key=key,
                    request_hash=request_hash,
                    appointment_id=appointment.id,
                    response_status=201,
                    created_at=now,
                    expires_at=now + self._ttl,
                )
            )
            uow.commit()
        _log.info("appointment_created", appointment_id=appointment.id, exam_count=len(exams))
        return appointment, False

    def _replay(self, uow: UnitOfWork, record: IdempotencyRecord, request_hash: str) -> Appointment:
        if record.request_hash != request_hash:
            _log.info("idempotency_conflict", appointment_id=record.appointment_id)
            raise IdempotencyKeyReuseError
        appointment = uow.appointments.get(record.appointment_id)
        if appointment is None:  # pragma: no cover - the foreign key makes this impossible
            raise RuntimeError(f"idempotency key points to missing {record.appointment_id}")
        _log.info("idempotent_replay", appointment_id=appointment.id)
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
