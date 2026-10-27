"""SQLAlchemy implementations of the domain storage protocols.

Statements are executed explicitly (Core-style through the session) rather
than through the ORM identity map, so each call hits the database when it is
made and the order of writes is the order of the code.
"""

from datetime import UTC
from types import TracebackType
from typing import Self

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from scheduler_api.domain.models import Appointment, ExamRef
from scheduler_api.infra.orm import AppointmentExamRow, AppointmentRow


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
            select(
                AppointmentRow.id,
                AppointmentRow.status,
                AppointmentRow.patient_ref,
                AppointmentRow.source_document_sha256,
                AppointmentRow.requested_by,
                AppointmentRow.created_at,
            ).where(AppointmentRow.id == appointment_id)
        ).one_or_none()
        if row is None:
            return None
        exams = self._session.execute(
            select(AppointmentExamRow.code, AppointmentExamRow.name)
            .where(AppointmentExamRow.appointment_id == appointment_id)
            .order_by(AppointmentExamRow.code)
        ).all()
        return Appointment(
            id=row.id,
            status=row.status,
            patient_ref=row.patient_ref,
            exams=tuple(ExamRef(code=e.code, name=e.name) for e in exams),
            source_document_sha256=row.source_document_sha256,
            requested_by=row.requested_by,
            # The driver returns the session time zone; the API always speaks UTC.
            created_at=row.created_at.astimezone(UTC),
        )


class SqlUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> Self:
        self._session = self._session_factory()
        self.appointments = SqlAppointmentRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Rolls back anything not committed, then returns the connection to the pool.
        self._session.close()

    def commit(self) -> None:
        self._session.commit()
