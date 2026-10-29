import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from scheduler_api.api.deps import require_api_key
from scheduler_api.api.errors import ApiError, FieldError

pytestmark = pytest.mark.unit

type LogRecords = Callable[[], list[dict[str, Any]]]

PROBLEM = "application/problem+json"
SENSITIVE = "Maria-Sentinela-da-Silva"


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    label: str


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    @app.post("/_test/payload")
    def payload(body: Payload) -> dict[str, int]:
        return {"count": body.count}

    @app.get("/_test/api-error")
    def api_error() -> None:
        raise ApiError(
            409,
            "something_conflicting",
            "Something conflicting",
            "It conflicts.",
            errors=[FieldError(loc=["body", "x"], code="dup", msg="duplicated")],
            headers={"Retry-After": "3"},
        )

    @app.get("/_test/boom")
    def boom() -> None:
        raise RuntimeError(f"internal detail {SENSITIVE}")

    @app.get("/_test/boom-mid-stream")
    def boom_mid_stream() -> StreamingResponse:
        def chunks() -> Iterator[bytes]:
            yield b"partial"
            raise RuntimeError("after headers")

        return StreamingResponse(chunks())

    @app.get("/_test/protected", dependencies=[Depends(require_api_key)])
    def protected() -> None:
        return None

    with TestClient(app) as client:
        yield client


def _problem(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM
    body: dict[str, Any] = response.json()
    assert body["type"] == f"https://errors.lab-agent.example/{code}"
    assert body["status"] == status
    assert body["request_id"] == response.headers["X-Request-ID"]
    return body


def test_unknown_route_is_a_problem(client: TestClient) -> None:
    response = client.get("/nope")

    body = _problem(response, 404, "not_found")
    assert body["title"] == "Not Found"
    assert body["instance"] == "/nope"
    assert "detail" not in body
    assert "errors" not in body


def test_method_not_allowed_keeps_allow_header(client: TestClient) -> None:
    response = client.post("/health/live")

    _problem(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "GET"


def test_missing_api_key_is_a_problem(client: TestClient) -> None:
    response = client.get("/_test/protected")

    body = _problem(response, 401, "unauthorized")
    assert body["detail"] == "Missing or invalid API key."


def test_validation_errors_list_each_field(client: TestClient) -> None:
    response = client.post(
        "/_test/payload", json={"count": "not-a-number", "patient_name": SENSITIVE}
    )

    body = _problem(response, 422, "validation_error")
    assert body["title"] == "Request validation failed"
    assert body["detail"] == "3 validation error(s)."
    assert sorted((e["loc"], e["code"]) for e in body["errors"]) == [
        (["body", "count"], "int_parsing"),
        (["body", "label"], "missing"),
        (["body", "patient_name"], "extra_forbidden"),
    ]
    assert all(set(e) == {"loc", "code", "msg"} for e in body["errors"])


def test_validation_errors_never_echo_input(client: TestClient) -> None:
    response = client.post("/_test/payload", json={"count": SENSITIVE, "label": 1})

    assert response.status_code == 422
    assert SENSITIVE not in response.text


def test_malformed_json_is_a_validation_problem(client: TestClient) -> None:
    response = client.post(
        "/_test/payload",
        content=f'{{"count": 1, "label": "{SENSITIVE}"',
        headers={"content-type": "application/json"},
    )

    body = _problem(response, 422, "validation_error")
    assert body["errors"][0]["code"] == "json_invalid"
    assert SENSITIVE not in response.text


def test_api_error_is_rendered_as_given(client: TestClient) -> None:
    response = client.get("/_test/api-error")

    body = _problem(response, 409, "something_conflicting")
    assert body == {
        "type": "https://errors.lab-agent.example/something_conflicting",
        "title": "Something conflicting",
        "status": 409,
        "detail": "It conflicts.",
        "instance": "/_test/api-error",
        "request_id": response.headers["X-Request-ID"],
        "errors": [{"loc": ["body", "x"], "code": "dup", "msg": "duplicated"}],
    }
    assert response.headers["Retry-After"] == "3"


def test_unexpected_error_is_a_generic_500(client: TestClient) -> None:
    response = client.get("/_test/boom", headers={"X-Request-ID": "req-boom-1"})

    body = _problem(response, 500, "internal_error")
    assert body["request_id"] == "req-boom-1"
    assert "internal detail" not in response.text
    assert SENSITIVE not in response.text
    assert "RuntimeError" not in response.text


def test_unexpected_error_logs_stack_trace_with_request_id(
    client: TestClient, log_records: LogRecords
) -> None:
    client.get("/_test/boom", headers={"X-Request-ID": "req-boom-2"})

    [error] = [r for r in log_records() if r["event"] == "unhandled_error"]
    assert error["request_id"] == "req-boom-2"
    assert error["level"] == "error"
    assert error["route"] == "/_test/boom"
    assert "RuntimeError" in error["exception"]
    [request] = [r for r in log_records() if r["event"] == "request"]
    assert request["status"] == 500


def test_unexpected_error_does_not_log_request_body(
    client: TestClient, log_records: LogRecords
) -> None:
    client.post("/_test/payload", json={"count": 1, "label": SENSITIVE})
    client.post("/_test/payload", json={"count": "x", "label": SENSITIVE})

    assert SENSITIVE not in json.dumps(
        [r for r in log_records() if not r["event"].startswith("HTTP Request:")]
    )


def test_error_after_response_started_is_reraised(client: TestClient) -> None:
    with pytest.raises(RuntimeError, match="after headers"):
        client.get("/_test/boom-mid-stream")
