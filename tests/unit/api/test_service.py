import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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
    IdempotencyKeyReuseError,
    IdempotencyKeyTakenError,
    IdempotencyRecord,
    canonical_request_hash,
)
from scheduler_api.domain.services import AppointmentService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 26, 22, 48, tzinfo=UTC)
SHA = "9b1e" + "0" * 60
TTL = timedelta(hours=24)
KEY = "key-0001"
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

    def list_recent(self, limit: int, patient_ref: str | None = None) -> list[Appointment]:
        matching = [a for a in self._store.values() if patient_ref in (None, a.patient_ref)]
        matching.sort(key=lambda a: (a.created_at, a.id), reverse=True)
        return matching[:limit]


class FakeIdempotency:
    """Mimics the primary key: adding a key that is already committed fails."""

    def __init__(self, db: "FakeDatabase") -> None:
        self._db = db
        self.pending: dict[str, IdempotencyRecord] = {}
        self.deleted: set[str] = set()

    def get(self, key: str) -> IdempotencyRecord | None:
        record = self._db.keys.get(key)
        if record is None:
            self._db.on_miss()
        return record

    def add(self, record: IdempotencyRecord) -> None:
        if self._db.failing_adds:
            self._db.failing_adds -= 1
            raise IdempotencyKeyTakenError(record.key)
        if record.key in self._db.keys and record.key not in self.deleted:
            raise IdempotencyKeyTakenError(record.key)
        self.pending[record.key] = record

    def delete_expired(self, key: str, now: datetime) -> None:
        if self._db.keys[key].expires_at <= now:
            self.deleted.add(key)


class FakeUnitOfWork:
    def __init__(self, db: "FakeDatabase") -> None:
        self._db = db
        self.appointments = FakeAppointments(db.appointments)
        self.idempotency = FakeIdempotency(db)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.appointments.pending.clear()
        self.idempotency.pending.clear()

    def commit(self) -> None:
        for key in self.idempotency.deleted:
            del self._db.keys[key]
        self._db.appointments.update(self.appointments.pending)
        self._db.keys.update(self.idempotency.pending)
        self.appointments.pending.clear()
        self.idempotency.pending.clear()


class FakeDatabase:
    def __init__(self) -> None:
        self.appointments: dict[str, Appointment] = {}
        self.keys: dict[str, IdempotencyRecord] = {}
        self.races: list[Callable[[], None]] = []
        """Run once each when a key lookup misses: a concurrent request committing first."""
        self.failing_adds = 0
        """Number of key inserts that fail as taken, without any record appearing."""

    def on_miss(self) -> None:
        if self.races:
            self.races.pop(0)()


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def store(db: FakeDatabase) -> dict[str, Appointment]:
    return db.appointments


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(db: FakeDatabase, clock: Clock) -> AppointmentService:
    ids = iter(f"apt_{n:026d}" for n in range(1, 100))
    return AppointmentService(
        lambda: FakeUnitOfWork(db),
        CATALOG,
        idempotency_ttl=TTL,
        clock=clock,
        new_id=lambda: next(ids),
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
    created, replayed = service.create(
        request(("LAB-10012", "Glicemia de jejum"), ("LAB-10001", "Hemograma completo")), KEY
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
    assert not replayed
    assert store == {created.id: created}


def test_names_are_compared_normalized_and_stored_as_in_catalog(
    service: AppointmentService,
) -> None:
    created, _ = service.create(request(("LAB-10003", "  CONTAGEM de reticulocitos. ")), KEY)

    assert created.exams == (ExamRef("LAB-10003", "Contagem de reticulócitos"),)


def test_unknown_code_is_rejected_without_writing(
    service: AppointmentService, store: dict[str, Appointment], db: FakeDatabase
) -> None:
    with pytest.raises(CatalogValidationError) as info:
        service.create(
            request(("LAB-10001", "Hemograma completo"), ("LAB-99999", "Inventado")), KEY
        )

    [issue] = info.value.issues
    assert (issue.index, issue.field, issue.kind) == (1, "code", "unknown_exam_code")
    assert issue.message == "LAB-99999 not found"
    assert store == {}
    assert db.keys == {}


def test_name_mismatch_is_rejected(service: AppointmentService) -> None:
    with pytest.raises(CatalogValidationError) as info:
        service.create(request(("LAB-10001", "Glicemia de jejum")), KEY)

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
            ),
            KEY,
        )

    assert [(i.index, i.kind) for i in info.value.issues] == [
        (0, "exam_name_mismatch"),
        (2, "unknown_exam_code"),
    ]
    assert info.value.count("unknown_exam_code") == 1


def test_get_returns_stored_appointment(service: AppointmentService) -> None:
    created, _ = service.create(request(("LAB-10001", "Hemograma completo")), KEY)

    assert service.get(created.id) == created


def test_get_missing_raises(service: AppointmentService) -> None:
    with pytest.raises(AppointmentNotFoundError) as info:
        service.get("apt_" + "9" * 26)

    assert info.value.appointment_id == "apt_" + "9" * 26


def test_list_recent_is_newest_first(service: AppointmentService, clock: Clock) -> None:
    first, _ = service.create(request(("LAB-10001", "Hemograma completo")), "key-a")
    clock.now = NOW + timedelta(minutes=1)
    second, _ = service.create(request(("LAB-10001", "Hemograma completo")), "key-b")

    assert service.list_recent(20) == [second, first]
    assert service.list_recent(1) == [second]


def test_list_recent_filters_by_patient(service: AppointmentService) -> None:
    mine, _ = service.create(request(("LAB-10001", "Hemograma completo")), "key-a")
    other = AppointmentRequest(
        "pt_AAAAAAAAAAAAAAAAAAAAAAAAAA", (ExamRef("LAB-10001", "Hemograma completo"),), SHA, "x"
    )
    service.create(other, "key-b")

    assert service.list_recent(20, mine.patient_ref) == [mine]
    assert service.list_recent(20, "pt_" + "Z" * 26) == []


def test_default_ids_are_prefixed_ulids() -> None:
    db = FakeDatabase()
    service = AppointmentService(lambda: FakeUnitOfWork(db), CATALOG)

    first, _ = service.create(request(("LAB-10001", "Hemograma completo")), "key-a")
    second, _ = service.create(request(("LAB-10001", "Hemograma completo")), "key-b")

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


# Idempotency


HEMOGRAMA = ("LAB-10001", "Hemograma completo")
GLICEMIA = ("LAB-10012", "Glicemia de jejum")


def test_records_the_key_with_hash_and_ttl(service: AppointmentService, db: FakeDatabase) -> None:
    created, _ = service.create(request(HEMOGRAMA), KEY)

    assert db.keys == {
        KEY: IdempotencyRecord(
            key=KEY,
            request_hash=canonical_request_hash(request(HEMOGRAMA)),
            appointment_id=created.id,
            response_status=201,
            created_at=NOW,
            expires_at=NOW + TTL,
        )
    }


def test_same_key_and_body_replays(service: AppointmentService, db: FakeDatabase) -> None:
    first, first_replayed = service.create(request(HEMOGRAMA, GLICEMIA), KEY)
    second, second_replayed = service.create(request(HEMOGRAMA, GLICEMIA), KEY)

    assert (first_replayed, second_replayed) == (False, True)
    assert second == first
    assert len(db.appointments) == 1


def test_exam_order_does_not_matter_for_replay(service: AppointmentService) -> None:
    first, _ = service.create(request(HEMOGRAMA, GLICEMIA), KEY)
    second, replayed = service.create(request(GLICEMIA, HEMOGRAMA), KEY)

    assert replayed
    assert second.id == first.id


@pytest.mark.parametrize(
    "changed",
    [
        request(HEMOGRAMA),
        request(HEMOGRAMA, ("LAB-10012", "GLICEMIA DE JEJUM")),
        AppointmentRequest(
            "pt_AAAAAAAAAAAAAAAAAAAAAAAAAA",
            (ExamRef(*HEMOGRAMA), ExamRef(*GLICEMIA)),
            SHA,
            "exam_scheduling_agent",
        ),
        AppointmentRequest(
            "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
            (ExamRef(*HEMOGRAMA), ExamRef(*GLICEMIA)),
            "f" * 64,
            "exam_scheduling_agent",
        ),
        AppointmentRequest(
            "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
            (ExamRef(*HEMOGRAMA), ExamRef(*GLICEMIA)),
            SHA,
            "other_system",
        ),
    ],
    ids=["fewer-exams", "name-spelling", "patient", "document", "requester"],
)
def test_same_key_with_different_body_conflicts(
    service: AppointmentService, db: FakeDatabase, changed: AppointmentRequest
) -> None:
    service.create(request(HEMOGRAMA, GLICEMIA), KEY)

    with pytest.raises(IdempotencyKeyReuseError):
        service.create(changed, KEY)
    assert len(db.appointments) == 1


def test_different_keys_create_separate_appointments(
    service: AppointmentService, db: FakeDatabase
) -> None:
    first, _ = service.create(request(HEMOGRAMA), "key-0001")
    second, replayed = service.create(request(HEMOGRAMA), "key-0002")

    assert not replayed
    assert first.id != second.id
    assert len(db.appointments) == 2


def test_expired_key_creates_a_new_appointment(
    service: AppointmentService, db: FakeDatabase, clock: Clock
) -> None:
    first, _ = service.create(request(HEMOGRAMA), KEY)
    clock.now = NOW + TTL

    second, replayed = service.create(request(HEMOGRAMA), KEY)

    assert not replayed
    assert second.id != first.id
    assert db.keys[KEY].appointment_id == second.id
    assert db.keys[KEY].expires_at == NOW + 2 * TTL


def test_expired_key_accepts_a_different_body(service: AppointmentService, clock: Clock) -> None:
    service.create(request(HEMOGRAMA), KEY)
    clock.now = NOW + TTL + timedelta(seconds=1)

    _, replayed = service.create(request(GLICEMIA), KEY)

    assert not replayed


def test_key_just_before_expiry_still_replays(service: AppointmentService, clock: Clock) -> None:
    service.create(request(HEMOGRAMA), KEY)
    clock.now = NOW + TTL - timedelta(microseconds=1)

    _, replayed = service.create(request(HEMOGRAMA), KEY)

    assert replayed


def _concurrent_commit(
    db: FakeDatabase, body: AppointmentRequest, appointment_id: str = "apt_" + "W" * 26
) -> Callable[[], None]:
    """A request with the same key that commits between our read and our insert."""

    def commit() -> None:
        winner = Appointment(
            appointment_id, "requested", body.patient_ref, body.exams, SHA, body.requested_by, NOW
        )
        db.appointments[winner.id] = winner
        db.keys[KEY] = IdempotencyRecord(
            KEY, canonical_request_hash(body), winner.id, 201, NOW, NOW + TTL
        )

    return commit


def test_race_on_insert_rereads_and_replays(service: AppointmentService, db: FakeDatabase) -> None:
    db.races.append(_concurrent_commit(db, request(HEMOGRAMA)))

    appointment, replayed = service.create(request(HEMOGRAMA), KEY)

    assert replayed
    assert appointment.id == "apt_" + "W" * 26
    assert list(db.appointments) == [appointment.id]


def test_race_with_different_body_conflicts(service: AppointmentService, db: FakeDatabase) -> None:
    db.races.append(_concurrent_commit(db, request(GLICEMIA)))

    with pytest.raises(IdempotencyKeyReuseError):
        service.create(request(HEMOGRAMA), KEY)


def test_second_race_in_a_row_is_not_retried(service: AppointmentService, db: FakeDatabase) -> None:
    db.failing_adds = 2

    with pytest.raises(IdempotencyKeyTakenError):
        service.create(request(HEMOGRAMA), KEY)
    assert db.appointments == {}


def test_single_failed_insert_is_retried(service: AppointmentService, db: FakeDatabase) -> None:
    db.failing_adds = 1

    _, replayed = service.create(request(HEMOGRAMA), KEY)

    assert not replayed
    assert len(db.appointments) == 1


def test_canonical_hash_is_independent_of_exam_order() -> None:
    assert canonical_request_hash(request(HEMOGRAMA, GLICEMIA)) == canonical_request_hash(
        request(GLICEMIA, HEMOGRAMA)
    )


def test_canonical_hash_format_is_pinned() -> None:
    expected_json = (
        '{"exams":[{"code":"LAB-10001","name":"Hemograma completo"},'
        '{"code":"LAB-10012","name":"Glicemia de jejum"}],'
        '"patient_ref":"pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",'
        '"requested_by":"exam_scheduling_agent",'
        f'"source_document_sha256":"{SHA}"}}'
    )

    assert (
        canonical_request_hash(request(GLICEMIA, HEMOGRAMA))
        == hashlib.sha256(expected_json.encode()).hexdigest()
    )


def test_canonical_hash_keeps_unicode_as_utf8() -> None:
    body = request(("LAB-10003", "Contagem de reticulócitos"))
    expected = (
        '{"exams":[{"code":"LAB-10003","name":"Contagem de reticulócitos"}],'
        '"patient_ref":"pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",'
        '"requested_by":"exam_scheduling_agent",'
        f'"source_document_sha256":"{SHA}"}}'
    )

    assert canonical_request_hash(body) == hashlib.sha256(expected.encode()).hexdigest()
