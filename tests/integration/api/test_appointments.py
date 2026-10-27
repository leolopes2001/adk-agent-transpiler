import io
from datetime import datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, text

from scheduler_api import log

pytestmark = pytest.mark.integration

PROBLEM = "application/problem+json"
PATIENT = "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"
SHA = "9b1e" + "0" * 60


def payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "patient_ref": PATIENT,
        "exams": [
            {"code": "LAB-10012", "name": "Glicemia de jejum"},
            {"code": "LAB-10001", "name": "Hemograma completo"},
        ],
        "source_document_sha256": SHA,
        "requested_by": "exam_scheduling_agent",
    }
    body.update(overrides)
    return body


async def post(client: httpx.AsyncClient, body: dict[str, Any], key: str = "key-0001") -> Any:
    return await client.post("/v1/appointments", json=body, headers={"Idempotency-Key": key})


def _problem(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM
    body: dict[str, Any] = response.json()
    assert body["type"] == f"https://errors.lab-agent.example/{code}"
    return body


async def test_create_returns_201_with_location(client: httpx.AsyncClient) -> None:
    response = await post(client, payload())

    assert response.status_code == 201, response.text
    body = response.json()
    assert response.headers["Location"] == f"/v1/appointments/{body['appointment_id']}"
    assert body["appointment_id"].startswith("apt_")
    assert body["status"] == "requested"
    assert body["patient_ref"] == PATIENT
    assert body["exams"] == [
        {"code": "LAB-10001", "name": "Hemograma completo"},
        {"code": "LAB-10012", "name": "Glicemia de jejum"},
    ]
    assert body["source_document_sha256"] == SHA
    assert body["requested_by"] == "exam_scheduling_agent"
    assert body["created_at"].endswith("Z")
    assert datetime.fromisoformat(body["created_at"]).utcoffset().total_seconds() == 0


async def test_get_returns_what_was_created(client: httpx.AsyncClient) -> None:
    created = await post(client, payload())

    fetched = await client.get(created.headers["Location"])

    assert fetched.status_code == 200
    assert fetched.json() == created.json()


async def test_rows_are_persisted(client: httpx.AsyncClient, db: Engine) -> None:
    created = await post(client, payload(exams=[{"code": "LAB-10003", "name": "reticulocitos"}]))
    assert created.status_code == 422

    response = await post(
        client, payload(exams=[{"code": "LAB-10003", "name": "CONTAGEM DE RETICULOCITOS"}])
    )
    assert response.status_code == 201

    with db.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT a.status, e.code, e.name FROM appointments a JOIN appointment_exams e"
                " ON e.appointment_id = a.id"
            )
        ).all()
    # The catalog name is stored, not the spelling the client used.
    assert [tuple(r) for r in rows] == [("requested", "LAB-10003", "Contagem de reticulócitos")]


async def test_unknown_code_is_422(client: httpx.AsyncClient, db: Engine) -> None:
    body = payload(
        exams=[
            {"code": "LAB-10001", "name": "Hemograma completo"},
            {"code": "LAB-99999", "name": "Exame inventado"},
        ]
    )

    problem = _problem(await post(client, body), 422, "unknown_exam_code")

    assert problem["title"] == "Unknown exam code"
    assert problem["detail"] == "1 exam code(s) do not exist in the catalog."
    assert problem["instance"] == "/v1/appointments"
    assert problem["errors"] == [
        {
            "loc": ["body", "exams", 1, "code"],
            "code": "unknown_exam_code",
            "msg": "LAB-99999 not found",
        }
    ]
    with db.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM appointments")).scalar_one() == 0


async def test_name_mismatch_is_422(client: httpx.AsyncClient) -> None:
    body = payload(exams=[{"code": "LAB-10001", "name": "Glicemia de jejum"}])

    problem = _problem(await post(client, body), 422, "exam_name_mismatch")

    [error] = problem["errors"]
    assert error["loc"] == ["body", "exams", 0, "name"]
    assert error["code"] == "exam_name_mismatch"


@pytest.mark.parametrize(
    ("overrides", "loc", "code"),
    [
        ({"patient_name": "Fulano"}, ["body", "patient_name"], "extra_forbidden"),
        ({"patient_ref": "12345678900"}, ["body", "patient_ref"], "string_pattern_mismatch"),
        ({"exams": []}, ["body", "exams"], "too_short"),
        (
            {"exams": [{"code": f"LAB-{10001 + i}", "name": "x"} for i in range(31)]},
            ["body", "exams"],
            "too_long",
        ),
        (
            {"exams": [{"code": "LAB-10001", "name": "Hemograma completo"}] * 2},
            ["body", "exams"],
            "duplicate_exam_code",
        ),
        ({"exams": [{"code": "lab-10001", "name": "x"}]}, ["body", "exams", 0, "code"], None),
        (
            {"exams": [{"code": "LAB-10001", "name": "x", "cpf": "1"}]},
            ["body", "exams", 0, "cpf"],
            "extra_forbidden",
        ),
        ({"source_document_sha256": "A" * 64}, ["body", "source_document_sha256"], None),
        ({"requested_by": "Agent X"}, ["body", "requested_by"], None),
        ({"status": "scheduled"}, ["body", "status"], "extra_forbidden"),
    ],
    ids=[
        "patient-name",
        "patient-ref",
        "no-exams",
        "too-many-exams",
        "duplicate-code",
        "code-format",
        "exam-extra-field",
        "sha-uppercase",
        "requested-by",
        "client-status",
    ],
)
async def test_contract_violations_are_422(
    client: httpx.AsyncClient, overrides: dict[str, Any], loc: list[Any], code: str | None
) -> None:
    problem = _problem(await post(client, payload(**overrides)), 422, "validation_error")

    matching = [e for e in problem["errors"] if e["loc"] == loc]
    assert matching, problem["errors"]
    if code is not None:
        assert matching[0]["code"] == code


async def test_get_unknown_appointment_is_404(client: httpx.AsyncClient) -> None:
    problem = _problem(
        await client.get("/v1/appointments/apt_01JB0000000000000000000000"),
        404,
        "appointment_not_found",
    )

    assert problem["title"] == "Appointment not found"


async def test_get_malformed_appointment_id_is_422(client: httpx.AsyncClient) -> None:
    _problem(await client.get("/v1/appointments/1"), 422, "validation_error")


async def test_get_exam_from_catalog(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/exams/LAB-10001")

    assert response.status_code == 200
    assert response.json() == {"code": "LAB-10001", "name": "Hemograma completo"}


async def test_get_unknown_exam_is_404(client: httpx.AsyncClient) -> None:
    _problem(await client.get("/v1/exams/LAB-99999"), 404, "exam_not_found")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/appointments"),
        ("GET", "/v1/appointments/apt_01JB0000000000000000000000"),
        ("GET", "/v1/exams/LAB-10001"),
    ],
)
async def test_v1_requires_api_key(client: httpx.AsyncClient, method: str, path: str) -> None:
    response = await client.request(
        method,
        path,
        json=payload() if method == "POST" else None,
        headers={"X-API-Key": "wrong-key-0123456789abcdefghijklmnopq"},
    )

    _problem(response, 401, "unauthorized")


async def test_request_id_is_echoed_in_problem_and_header(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/exams/LAB-99999", headers={"X-Request-ID": "run-77"})

    assert response.headers["X-Request-ID"] == "run-77"
    assert response.json()["request_id"] == "run-77"


async def test_ready_checks_database_and_catalog(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok", "catalog": "ok"}}


async def test_logs_name_the_appointment_but_not_the_request(client: httpx.AsyncClient) -> None:
    stream = io.StringIO()
    log.configure_logging(stream=stream)

    response = await post(client, payload())

    output = stream.getvalue()
    assert '"appointment_created"' in output
    assert response.json()["appointment_id"] in output
    for value in (PATIENT, SHA, "Hemograma completo", "Glicemia de jejum"):
        assert value not in output
