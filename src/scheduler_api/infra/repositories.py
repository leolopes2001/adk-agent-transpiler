"""Core statements, not the ORM identity map: each call hits the database in code order."""

from collections import defaultdict
from datetime import UTC, datetime
from types import TracebackType
from typing import Self

from sqlalchemy import Row, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from scheduler_api.domain.models import (
    Appointment,
    ExamRef,
    IdempotencyKeyTakenError,
    IdempotencyRecord,
)
from scheduler_api.infra.orm import (
    IDEMPOTENCY_KEY_PK,
    AppointmentExamRow,
    AppointmentRow,
    IdempotencyKeyRow,
)


class SqlAppointmentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, appointment: Appointment) -> None:
        self._session.execute(
            insert(AppointmentRow).values(
                id=appointment.id,
                status=appointment.status,
                patient_ref=appointment.patient_ref,
                source_document_sha256=appointment.source_document_sha256,
                requested_by=appointment.requested_by,
                created_at=appointment.created_at,
            )
        )
        self._session.execute(
            insert(AppointmentExamRow),
            [
                {"appointment_id": appointment.id, "code": exam.code, "name": exam.name}
                for exam in appointment.exams
            ],
        )

    def get(self, appointment_id: str) -> Appointment | None:
        row = self._session.execute(
            _APPOINTMENT_COLUMNS.where(AppointmentRow.id == appointment_id)
        ).one_or_none()
        if row is None:
            return None

        exams = self._exams_by_appointment([appointment_id])
        return _appointment(row, exams.get(appointment_id, ()))

    def list_recent(self, limit: int, patient_ref: str | None = None) -> list[Appointment]:
        query = _APPOINTMENT_COLUMNS.order_by(
            AppointmentRow.created_at.desc(), AppointmentRow.id.desc()
        ).limit(limit)
        if patient_ref is not None:
            query = query.where(AppointmentRow.patient_ref == patient_ref)

        rows = self._session.execute(query).all()
        exams = self._exams_by_appointment([row.id for row in rows])
        return [_appointment(row, exams.get(row.id, ())) for row in rows]

    def _exams_by_appointment(self, appointment_ids: list[str]) -> dict[str, tuple[ExamRef, ...]]:
        if not appointment_ids:
            return {}

        exam_rows = self._session.execute(
            select(
                AppointmentExamRow.appointment_id, AppointmentExamRow.code, AppointmentExamRow.name
            )
            .where(AppointmentExamRow.appointment_id.in_(appointment_ids))
            .order_by(AppointmentExamRow.code)
        ).all()
        exams: dict[str, list[ExamRef]] = defaultdict(list)
        for exam in exam_rows:
            exams[exam.appointment_id].append(ExamRef(code=exam.code, name=exam.name))

        return {appointment_id: tuple(refs) for appointment_id, refs in exams.items()}


class SqlIdempotencyRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, key: str) -> IdempotencyRecord | None:
        row = self._session.execute(
            select(
                IdempotencyKeyRow.key,
                IdempotencyKeyRow.request_hash,
                IdempotencyKeyRow.appointment_id,
                IdempotencyKeyRow.response_status,
                IdempotencyKeyRow.created_at,
                IdempotencyKeyRow.expires_at,
            ).where(IdempotencyKeyRow.key == key)
        ).one_or_none()
        if row is None:
            return None

        return IdempotencyRecord(
            key=row.key,
            request_hash=row.request_hash,
            appointment_id=row.appointment_id,
            response_status=row.response_status,
            created_at=row.created_at.astimezone(UTC),
            expires_at=row.expires_at.astimezone(UTC),
        )

    def add(self, record: IdempotencyRecord) -> None:
        try:
            self._session.execute(
                insert(IdempotencyKeyRow).values(
                    key=record.key,
                    request_hash=record.request_hash,
                    appointment_id=record.appointment_id,
                    response_status=record.response_status,
                    created_at=record.created_at,
                    expires_at=record.expires_at,
                )
            )
        except IntegrityError as exc:
            if _violated_constraint_name(exc) == IDEMPOTENCY_KEY_PK:
                raise IdempotencyKeyTakenError(record.key) from exc

            raise

    def delete_expired(self, key: str, now: datetime) -> None:
        # Rechecked so the fresh record of a concurrent request that replaced the key is kept.
        self._session.execute(
            delete(IdempotencyKeyRow).where(
                IdempotencyKeyRow.key == key, IdempotencyKeyRow.expires_at <= now
            )
        )


class SqlUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> Self:
        self._session = self._session_factory()
        self.appointments = SqlAppointmentRepository(self._session)
        self.idempotency = SqlIdempotencyRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._session.close()

    def commit(self) -> None:
        self._session.commit()


_AppointmentColumns = Row[str, str, str, str, str, datetime]

_APPOINTMENT_COLUMNS = select(
    AppointmentRow.id,
    AppointmentRow.status,
    AppointmentRow.patient_ref,
    AppointmentRow.source_document_sha256,
    AppointmentRow.requested_by,
    AppointmentRow.created_at,
)


def _appointment(row: _AppointmentColumns, exams: tuple[ExamRef, ...]) -> Appointment:
    return Appointment(
        id=row.id,
        status=row.status,
        patient_ref=row.patient_ref,
        exams=exams,
        source_document_sha256=row.source_document_sha256,
        requested_by=row.requested_by,
        # The driver returns the session time zone; the API always speaks UTC.
        created_at=row.created_at.astimezone(UTC),
    )


def _violated_constraint_name(exc: IntegrityError) -> str | None:
    diagnostics = getattr(exc.orig, "diag", None)
    name = getattr(diagnostics, "constraint_name", None)
    return name if isinstance(name, str) else None
