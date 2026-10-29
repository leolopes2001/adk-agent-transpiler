"""Initial schema: appointments, their exams and idempotency keys.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "appointments",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("patient_ref", sa.Text(), nullable=False),
        sa.Column("source_document_sha256", sa.CHAR(64), nullable=False),
        sa.Column("requested_by", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("status IN ('requested')", name="ck_appointments_status"),
        sa.PrimaryKeyConstraint("id", name="pk_appointments"),
    )
    op.create_table(
        "appointment_exams",
        sa.Column("appointment_id", sa.Text(), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["appointment_id"],
            ["appointments.id"],
            name="fk_appointment_exams_appointment_id_appointments",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("appointment_id", "code", name="pk_appointment_exams"),
    )
    op.create_table(
        "idempotency_keys",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.CHAR(64), nullable=False),
        sa.Column("appointment_id", sa.Text(), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["appointment_id"],
            ["appointments.id"],
            name="fk_idempotency_keys_appointment_id_appointments",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("key", name="pk_idempotency_keys"),
    )
    op.create_index("ix_idempotency_keys_appointment_id", "idempotency_keys", ["appointment_id"])
    op.create_index("ix_idempotency_keys_expires_at", "idempotency_keys", ["expires_at"])


def downgrade() -> None:
    op.drop_table("idempotency_keys")
    op.drop_table("appointment_exams")
    op.drop_table("appointments")
