import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest

from scheduler_api.domain.catalog import Catalog, CatalogError, CatalogExam, normalize_name
from scheduler_api.domain.models import (
    Appointment,
    AppointmentNotFoundError,
    AppointmentRequest,
    CatalogValidationError,
    ExamRef,
)
from scheduler_api.domain.services import AppointmentService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 26, 22, 48, tzinfo=UTC)
SHA = "9b1e" + "0" * 60
CATALOG = Catalog(
    [
        CatalogExam("LAB-10001", "Hemograma completo"),
        CatalogExam("LAB-10012", "Glicemia de jejum"),
        CatalogExam("LAB-10003", "Contagem de reticulócitos"),
    ],
    version="test",
)


class FakeAppointments:
    def __init__(self, store: dict[str, Appointment]) -> None:
        self._store = store
        self.pending: dict[str, Appointment] = {}

    def add(self, appointment: Appointment) -> None:
        self.pending[appointment.id] = appointment

    def get(self, appointment_id: str) -> Appointment | None:
        return self.pending.get(appointment_id) or self._store.get(appointment_id)


class FakeUnitOfWork:
    def __init__(self, store: dict[str, Appointment]) -> None:
        self._store = store
        self.appointments = FakeAppointments(store)
        self.committed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.appointments.pending.clear()

    def commit(self) -> None:
        self._store.update(self.appointments.pending)
        self.appointments.pending.clear()
        self.committed = True


@pytest.fixture
def store() -> dict[str, Appointment]:
    return {}


@pytest.fixture
def service(store: dict[str, Appointment]) -> AppointmentService:
    ids = iter(f"apt_{n:026d}" for n in range(1, 100))
    return AppointmentService(
        lambda: FakeUnitOfWork(store), CATALOG, clock=lambda: NOW, new_id=lambda: next(ids)
    )


def request(*exams: tuple[str, str]) -> AppointmentRequest:
    return AppointmentRequest(
        patient_ref="pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
        exams=tuple(ExamRef(code, name) for code, name in exams),
        source_document_sha256=SHA,
        requested_by="exam_scheduling_agent",
    )


def test_create_records_a_requested_appointment(
    service: AppointmentService, store: dict[str, Appointment]
) -> None:
    created = service.create(
        request(("LAB-10012", "Glicemia de jejum"), ("LAB-10001", "Hemograma completo"))
    )

    assert created == Appointment(
        id="apt_" + "1".zfill(26),
        status="requested",
        patient_ref="pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
        exams=(
            ExamRef("LAB-10001", "Hemograma completo"),
            ExamRef("LAB-10012", "Glicemia de jejum"),
        ),
        source_document_sha256=SHA,
        requested_by="exam_scheduling_agent",
        created_at=NOW,
    )
    assert store == {created.id: created}


def test_names_are_compared_normalized_and_stored_as_in_catalog(
    service: AppointmentService,
) -> None:
    created = service.create(request(("LAB-10003", "  CONTAGEM de reticulocitos. ")))

    assert created.exams == (ExamRef("LAB-10003", "Contagem de reticulócitos"),)


def test_unknown_code_is_rejected_without_writing(
    service: AppointmentService, store: dict[str, Appointment]
) -> None:
    with pytest.raises(CatalogValidationError) as info:
        service.create(request(("LAB-10001", "Hemograma completo"), ("LAB-99999", "Inventado")))

    [issue] = info.value.issues
    assert (issue.index, issue.field, issue.kind) == (1, "code", "unknown_exam_code")
    assert issue.message == "LAB-99999 not found"
    assert store == {}


def test_name_mismatch_is_rejected(service: AppointmentService) -> None:
    with pytest.raises(CatalogValidationError) as info:
        service.create(request(("LAB-10001", "Glicemia de jejum")))

    [issue] = info.value.issues
    assert (issue.index, issue.field, issue.kind) == (0, "name", "exam_name_mismatch")
    assert "Hemograma completo" in issue.message


def test_reports_every_catalog_issue(service: AppointmentService) -> None:
    with pytest.raises(CatalogValidationError) as info:
        service.create(
            request(
                ("LAB-10001", "Outro nome"),
                ("LAB-10012", "Glicemia de jejum"),
                ("LAB-00000", "Nada"),
            )
        )

    assert [(i.index, i.kind) for i in info.value.issues] == [
        (0, "exam_name_mismatch"),
        (2, "unknown_exam_code"),
    ]
    assert info.value.count("unknown_exam_code") == 1


def test_get_returns_stored_appointment(service: AppointmentService) -> None:
    created = service.create(request(("LAB-10001", "Hemograma completo")))

    assert service.get(created.id) == created


def test_get_missing_raises(service: AppointmentService) -> None:
    with pytest.raises(AppointmentNotFoundError) as info:
        service.get("apt_" + "9" * 26)

    assert info.value.appointment_id == "apt_" + "9" * 26


def test_default_ids_are_prefixed_ulids() -> None:
    service = AppointmentService(lambda: FakeUnitOfWork({}), CATALOG)

    first = service.create(request(("LAB-10001", "Hemograma completo")))
    second = service.create(request(("LAB-10001", "Hemograma completo")))

    assert first.id.startswith("apt_")
    assert len(first.id) == 30
    assert first.id != second.id
    assert first.created_at.tzinfo is UTC


# Catalog


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Contagem de reticulócitos", "contagem de reticulocitos"),
        ("Tipagem sanguínea ABO e fator Rh", "TIPAGEM  SANGUINEA abo e fator rh"),
        ("Glicose pós-prandial", "glicose pos prandial"),
        ("Colesterol total e frações", "colesterol total e fracoes!"),
    ],
)
def test_normalize_name_ignores_case_accents_and_punctuation(a: str, b: str) -> None:
    assert normalize_name(a) == normalize_name(b)


def test_normalize_name_keeps_words_distinct() -> None:
    assert normalize_name("Glicemia de jejum") != normalize_name("Glicemia jejum")


def test_loads_project_catalog() -> None:
    catalog = Catalog.load(Path(__file__).parents[3] / "data" / "exams_catalog.json")

    assert len(catalog) == 120
    assert catalog.get("LAB-10001") == CatalogExam("LAB-10001", "Hemograma completo")
    assert catalog.version


def _write(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        {"catalog_version": "1"},
        {"exams": []},
        {"catalog_version": "1", "exams": []},
        {"catalog_version": "1", "exams": [{"code": "LAB-1"}]},
        {"catalog_version": "1", "exams": [{"code": 1, "name": "x"}]},
        {"catalog_version": "1", "exams": [{"code": "A", "name": "x"}, {"code": "A", "name": "y"}]},
    ],
    ids=["json", "no-exams", "no-version", "empty", "no-name", "bad-code", "duplicate"],
)
def test_rejects_malformed_catalog(tmp_path: Path, content: Any) -> None:
    with pytest.raises(CatalogError):
        Catalog.load(_write(tmp_path, content))


def test_rejects_missing_catalog(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="cannot read"):
        Catalog.load(tmp_path / "missing.json")
