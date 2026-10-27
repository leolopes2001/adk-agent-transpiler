import io
from datetime import timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, text

from scheduler_api import log

pytestmark = pytest.mark.integration

PROBLEM = "application/problem+json"
KEY = "v1-9b1e0000:pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"


def payload(*codes: str) -> dict[str, Any]:
    names = {
        "LAB-10001": "Hemograma completo",
        "LAB-10012": "Glicemia de jejum",
        "LAB-10003": "Contagem de reticulócitos",
    }
    return {
        "patient_ref": "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
        "exams": [{"code": c, "name": names[c]} for c in codes or ("LAB-10001", "LAB-10012")],
        "source_document_sha256": "9b1e" + "0" * 60,
        "requested_by": "exam_scheduling_agent",
    }


async def post(
    client: httpx.AsyncClient, body: dict[str, Any], key: str | None = KEY
) -> httpx.Response:
    headers = {} if key is None else {"Idempotency-Key": key}
    return await client.post("/v1/appointments", json=body, headers=headers)


def _count(db: Engine, table: str) -> int:
    with db.connect() as connection:
        count: int = connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()  # noqa: S608
    return count


async def test_replay_returns_200_with_the_original_appointment(
    client: httpx.AsyncClient, db: Engine
) -> None:
    created = await post(client, payload())
    replayed = await post(client, payload())

    assert created.status_code == 201
    assert "Idempotent-Replayed" not in created.headers
    assert replayed.status_code == 200
    assert replayed.headers["Idempotent-Replayed"] == "true"
    assert replayed.headers["Location"] == created.headers["Location"]
    assert replayed.json() == created.json()
    assert _count(db, "appointments") == 1


async def test_replay_ignores_exam_order(client: httpx.AsyncClient, db: Engine) -> None:
    created = await post(client, payload("LAB-10001", "LAB-10012"))
    replayed = await post(client, payload("LAB-10012", "LAB-10001"))

    assert replayed.status_code == 200
    assert replayed.json()["appointment_id"] == created.json()["appointment_id"]


async def test_same_key_with_different_body_is_409(client: httpx.AsyncClient, db: Engine) -> None:
    await post(client, payload("LAB-10001"))
    response = await post(client, payload("LAB-10001", "LAB-10003"))

    assert response.status_code == 409
    assert response.headers["content-type"] == PROBLEM
    problem = response.json()
    assert problem["type"] == "https://errors.lab-agent.example/idempotency_key_reuse"
    assert problem["title"] == "Idempotency key reuse"
    assert _count(db, "appointments") == 1


async def test_different_keys_create_two_appointments(
    client: httpx.AsyncClient, db: Engine
) -> None:
    first = await post(client, payload(), key="key-aaaa-1")
    second = await post(client, payload(), key="key-aaaa-2")

    assert (first.status_code, second.status_code) == (201, 201)
    assert first.json()["appointment_id"] != second.json()["appointment_id"]
    assert _count(db, "appointments") == 2


@pytest.mark.parametrize(
    ("key", "code"),
    [
        (None, "missing"),
        ("short", "string_too_short"),
        ("k" * 129, "string_too_long"),
        ("has space here", "string_pattern_mismatch"),
        ("chave/com/barra", "string_pattern_mismatch"),
    ],
    ids=["missing", "short", "long", "space", "slash"],
)
async def test_invalid_key_is_400(
    client: httpx.AsyncClient, db: Engine, key: str | None, code: str
) -> None:
    response = await post(client, payload(), key=key)

    assert response.status_code == 400
    assert response.headers["content-type"] == PROBLEM
    problem = response.json()
    assert problem["type"] == "https://errors.lab-agent.example/invalid_idempotency_key"
    [error] = problem["errors"]
    assert error["loc"] == ["header", "Idempotency-Key"]
    assert error["code"] == code
    assert _count(db, "appointments") == 0


async def test_invalid_key_wins_over_body_errors(client: httpx.AsyncClient) -> None:
    response = await post(client, {"patient_name": "Fulano"}, key=None)

    assert response.status_code == 400
    assert [e["loc"] for e in response.json()["errors"]] == [["header", "Idempotency-Key"]]


async def test_invalid_body_does_not_consume_the_key(client: httpx.AsyncClient) -> None:
    bad = payload()
    bad["exams"][0]["code"] = "LAB-99999"

    assert (await post(client, bad)).status_code == 422
    assert (await post(client, payload())).status_code == 201


async def test_key_is_stored_with_hash_and_ttl(client: httpx.AsyncClient, db: Engine) -> None:
    created = (await post(client, payload())).json()

    with db.connect() as connection:
        row = connection.execute(
            text("SELECT * FROM idempotency_keys WHERE key = :key"), {"key": KEY}
        ).one()
    assert row.appointment_id == created["appointment_id"]
    assert row.response_status == 201
    assert len(row.request_hash) == 64
    assert row.expires_at - row.created_at == timedelta(hours=24)


async def test_expired_key_creates_a_new_appointment(client: httpx.AsyncClient, db: Engine) -> None:
    first = (await post(client, payload())).json()
    with db.begin() as connection:
        connection.execute(
            text("UPDATE idempotency_keys SET expires_at = now() - interval '1 second'")
        )

    response = await post(client, payload())

    assert response.status_code == 201
    second = response.json()
    assert second["appointment_id"] != first["appointment_id"]
    assert _count(db, "appointments") == 2
    with db.connect() as connection:
        pointed = connection.execute(
            text("SELECT appointment_id FROM idempotency_keys WHERE key = :key"), {"key": KEY}
        ).scalar_one()
    assert pointed == second["appointment_id"]


async def test_replay_and_conflict_are_logged_without_body(
    client: httpx.AsyncClient,
) -> None:
    stream = io.StringIO()
    log.configure_logging(stream=stream)

    await post(client, payload())
    await post(client, payload())
    await post(client, payload("LAB-10003"))

    output = stream.getvalue()
    for event in ("appointment_created", "idempotent_replay", "idempotency_conflict"):
        assert f'"{event}"' in output
    assert "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA" not in output
    assert KEY not in output
