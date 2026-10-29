"""SchedulerClient <-> scheduler API OpenAPI document.

The document is generated from the API code (`create_app().openapi()`), so a
change on either side that breaks the other fails here: the client's request
must validate against the operation's schemas, and every response the
operation documents must map to the outcome the runtime expects.
"""

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import structlog
from jsonschema import Draft202012Validator

from agent_runtime.errors import TransientError
from agent_runtime.gateways import scheduler
from agent_runtime.gateways.scheduler import AppointmentRequest, SchedulerClient, SchedulerRejected
from scheduler_api import log, openapi

pytestmark = pytest.mark.contract

REQUEST = AppointmentRequest(
    patient_ref="pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
    exams=(("LAB-10001", "Hemograma completo"), ("LAB-10043", "TSH")),
    source_document_sha256=hashlib.sha256(b"doc").hexdigest(),
    requested_by="exam_scheduling_agent",
)
IDEMPOTENCY_KEY = hashlib.sha256(b"v1|doc|pt|LAB-10001").hexdigest()
REQUEST_ID = "2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11"


@pytest.fixture(scope="module")
def spec() -> Iterator[dict[str, Any]]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    loaded: dict[str, Any] = json.loads(openapi.render())
    yield loaded
    # create_app configures the API's logging; leave the process as it was.
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    log._handler = None
    structlog.reset_defaults()


@pytest.fixture(scope="module")
def operation(spec: dict[str, Any]) -> dict[str, Any]:
    op: dict[str, Any] = spec["paths"][scheduler.APPOINTMENTS_PATH]["post"]
    return op


def _validator(spec: dict[str, Any], schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator({**schema, "components": spec["components"]})


def _parameter(operation: dict[str, Any], name: str) -> dict[str, Any]:
    [param] = [p for p in operation["parameters"] if p["name"] == name and p["in"] == "header"]
    return param


def test_request_body_validates_against_the_operation_schema(
    spec: dict[str, Any], operation: dict[str, Any]
) -> None:
    schema = operation["requestBody"]["content"]["application/json"]["schema"]

    errors = list(_validator(spec, schema).iter_errors(REQUEST.payload()))

    assert errors == []


def test_client_headers_are_the_ones_the_operation_declares(
    spec: dict[str, Any], operation: dict[str, Any]
) -> None:
    [security] = operation["security"]
    [scheme_name] = security
    scheme = spec["components"]["securitySchemes"][scheme_name]
    idempotency = _parameter(operation, scheduler.IDEMPOTENCY_KEY_HEADER)

    assert (scheme["type"], scheme["in"], scheme["name"]) == (
        "apiKey",
        "header",
        scheduler.API_KEY_HEADER,
    )
    assert idempotency["required"] is True
    assert re.fullmatch(idempotency["schema"]["pattern"], IDEMPOTENCY_KEY)
    assert _parameter(operation, scheduler.REQUEST_ID_HEADER)["required"] is False


def _examples(response: dict[str, Any]) -> list[dict[str, Any]]:
    [media] = response["content"].values()
    if "example" in media:
        return [media["example"]]
    return [example["value"] for example in media["examples"].values()]


def _request_answered_by(body: dict[str, Any]) -> AppointmentRequest:
    """The request a success example answers; the client checks the body against it."""
    if "appointment_id" not in body:
        return REQUEST
    return AppointmentRequest(
        patient_ref=body["patient_ref"],
        exams=tuple((e["code"], e["name"]) for e in body["exams"]),
        source_document_sha256=body["source_document_sha256"],
        requested_by=body["requested_by"],
    )


async def _send(status: int, body: dict[str, Any], headers: dict[str, str]) -> Any:
    transport = httpx.MockTransport(lambda r: httpx.Response(status, json=body, headers=headers))
    client = SchedulerClient("http://api.local", "k" * 32, 5.0, transport=transport)
    try:
        return await client.create_appointment(
            _request_answered_by(body), idempotency_key=IDEMPOTENCY_KEY, request_id=REQUEST_ID
        )
    finally:
        await client.aclose()


async def test_every_documented_response_maps_to_an_outcome(operation: dict[str, Any]) -> None:
    responses = operation["responses"]

    assert set(responses) == {"200", "201", "400", "401", "409", "422", "500"}
    for status_text, response in responses.items():
        status = int(status_text)
        for body in _examples(response):
            if status == 201:
                result = await _send(status, body, {})
                assert (result.status, result.appointment_id) == (
                    "scheduled",
                    body["appointment_id"],
                )
            elif status == 200:
                assert scheduler.REPLAYED_HEADER in response["headers"]
                result = await _send(status, body, {scheduler.REPLAYED_HEADER: "true"})
                assert (result.status, result.appointment_id) == (
                    "replayed",
                    body["appointment_id"],
                )
            elif status >= 500:
                with pytest.raises(TransientError):
                    await _send(status, body, {})
            else:
                with pytest.raises(SchedulerRejected) as caught:
                    await _send(status, body, {})
                assert caught.value.title == body["title"]


def test_success_examples_validate_against_the_response_schema(
    spec: dict[str, Any], operation: dict[str, Any]
) -> None:
    created = operation["responses"]["201"]["content"]["application/json"]
    validator = _validator(spec, created["schema"])

    assert list(validator.iter_errors(created["example"])) == []
    assert scheduler.AppointmentOut.model_validate(created["example"])


def _resolved(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref is None:
        return schema
    resolved: dict[str, Any] = spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    return resolved


def test_client_checks_every_field_of_the_success_body(
    spec: dict[str, Any], operation: dict[str, Any]
) -> None:
    for status in ("200", "201"):
        media = operation["responses"][status]["content"]["application/json"]
        schema = _resolved(spec, media["schema"])
        client = scheduler.AppointmentOut.model_json_schema()

        assert set(client["properties"]) == set(schema["properties"])
        assert set(client["required"]) == set(schema["required"])
        for field in ("status", "appointment_id"):
            for keyword in ("const", "pattern"):
                assert client["properties"][field].get(keyword) == (
                    schema["properties"][field].get(keyword)
                ), (field, keyword)


@pytest.mark.parametrize(
    ("path", "method"),
    [("/v1/appointments", "get"), ("/v1/appointments/{appointment_id}", "get")],
)
def test_read_routes_return_the_model_the_client_parses(
    spec: dict[str, Any], operation: dict[str, Any], path: str, method: str
) -> None:
    created = operation["responses"]["201"]["content"]["application/json"]["schema"]
    read = spec["paths"][path][method]["responses"]["200"]["content"]["application/json"]
    schema = read["schema"].get("items", read["schema"])
    example = read["example"][0] if isinstance(read["example"], list) else read["example"]

    assert schema == created
    assert scheduler.AppointmentOut.model_validate(example)
