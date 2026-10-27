"""Idempotency under real concurrency, against Postgres.

Sync endpoints run in a thread pool, so concurrent requests through the ASGI
transport overlap in the database for real. The forced-race tests make every
request read the key before any of them inserts it, so all but one must hit
the primary key and go through the re-read path.
"""

import asyncio
import io
import json
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, text

from scheduler_api import log
from scheduler_api.domain.models import IdempotencyRecord
from scheduler_api.infra.repositories import SqlIdempotencyRepository

pytestmark = pytest.mark.integration

N = 10
KEY = "concurrency-key-0001"


def payload(code: str = "LAB-10001", name: str = "Hemograma completo") -> dict[str, Any]:
    return {
        "patient_ref": "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
        "exams": [{"code": code, "name": name}],
        "source_document_sha256": "9b1e" + "0" * 60,
        "requested_by": "exam_scheduling_agent",
    }


async def burst(
    client: httpx.AsyncClient, bodies: list[dict[str, Any]], keys: list[str]
) -> list[httpx.Response]:
    return list(
        await asyncio.gather(
            *(
                client.post("/v1/appointments", json=body, headers={"Idempotency-Key": key})
                for body, key in zip(bodies, keys, strict=True)
            )
        )
    )


def _count(db: Engine, table: str) -> int:
    with db.connect() as connection:
        count: int = connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()  # noqa: S608
    return count


@pytest.fixture
def all_read_before_insert(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Holds the first N key lookups until all N have happened; returns the lookup count."""
    barrier = threading.Barrier(N, timeout=15)
    lock = threading.Lock()
    lookups = 0
    original = SqlIdempotencyRepository.get

    def get(self: SqlIdempotencyRepository, key: str) -> IdempotencyRecord | None:
        nonlocal lookups
        record = original(self, key)
        with lock:
            lookups += 1
            first_round = lookups <= N
        if first_round:
            barrier.wait()
        return record

    monkeypatch.setattr(SqlIdempotencyRepository, "get", get)
    return lambda: lookups


def _assert_single_appointment(responses: list[httpx.Response], db: Engine) -> None:
    statuses = Counter(r.status_code for r in responses)
    assert statuses == {201: 1, 200: N - 1}, [r.text for r in responses]
    assert len({r.json()["appointment_id"] for r in responses}) == 1
    assert all(
        r.headers["Idempotent-Replayed"] == "true" for r in responses if r.status_code == 200
    )
    assert _count(db, "appointments") == 1
    assert _count(db, "appointment_exams") == 1
    assert _count(db, "idempotency_keys") == 1


async def test_same_key_in_parallel_creates_one_appointment(
    client: httpx.AsyncClient, db: Engine
) -> None:
    responses = await burst(client, [payload()] * N, [KEY] * N)

    _assert_single_appointment(responses, db)


async def test_forced_race_goes_through_the_primary_key(
    client: httpx.AsyncClient, db: Engine, all_read_before_insert: Callable[[], int]
) -> None:
    stream = io.StringIO()
    log.configure_logging(stream=stream)

    responses = await burst(client, [payload()] * N, [KEY] * N)

    _assert_single_appointment(responses, db)
    events = Counter(json.loads(line)["event"] for line in stream.getvalue().splitlines())
    # Every request read "no key"; the N - 1 losers hit the constraint and re-read once.
    assert events["idempotency_race_retry"] == N - 1
    assert events["idempotent_replay"] == N - 1
    assert events["appointment_created"] == 1
    assert all_read_before_insert() == N + (N - 1)


async def test_forced_race_with_two_bodies_conflicts_for_the_loser_body(
    client: httpx.AsyncClient, db: Engine, all_read_before_insert: Callable[[], int]
) -> None:
    first, second = payload(), payload("LAB-10012", "Glicemia de jejum")
    bodies = [first, second] * (N // 2)

    responses = await burst(client, bodies, [KEY] * N)

    [winner] = [r for r in responses if r.status_code == 201]
    winner_codes = winner.json()["exams"][0]["code"]
    for body, response in zip(bodies, responses, strict=True):
        if response is winner:
            continue
        same_body = body["exams"][0]["code"] == winner_codes
        assert response.status_code == (200 if same_body else 409), response.text
    assert _count(db, "appointments") == 1


async def test_different_keys_in_parallel_create_one_appointment_each(
    client: httpx.AsyncClient, db: Engine
) -> None:
    responses = await burst(client, [payload()] * N, [f"parallel-key-{i:04d}" for i in range(N)])

    assert [r.status_code for r in responses] == [201] * N
    assert len({r.json()["appointment_id"] for r in responses}) == N
    assert _count(db, "appointments") == N
    assert _count(db, "idempotency_keys") == N
