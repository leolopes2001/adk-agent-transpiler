from datetime import datetime

from sqlalchemy import (
    CHAR,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

IDEMPOTENCY_KEY_PK = "pk_idempotency_keys"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class AppointmentRow(Base):
    __tablename__ = "appointments"
    __table_args__ = (CheckConstraint("status IN ('requested')", name="status"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text)
    patient_ref: Mapped[str] = mapped_column(Text)
    source_document_sha256: Mapped[str] = mapped_column(CHAR(64))
    requested_by: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AppointmentExamRow(Base):
    __tablename__ = "appointment_exams"

    appointment_id: Mapped[str] = mapped_column(
        Text, ForeignKey("appointments.id", ondelete="CASCADE"), primary_key=True
    )
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)


class IdempotencyKeyRow(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(CHAR(64))
    appointment_id: Mapped[str] = mapped_column(
        Text, ForeignKey("appointments.id", ondelete="CASCADE"), index=True
    )
    response_status: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
