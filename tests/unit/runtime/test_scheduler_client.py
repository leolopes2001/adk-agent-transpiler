import json
from collections.abc import Callable

import httpx
import pytest

from agent_runtime.errors import PermanentError, TransientError
from agent_runtime.gateways.scheduler import (
    AppointmentRequest,
    AppointmentResult,
    SchedulerClient,
    SchedulerRejected,
)

pytestmark = pytest.mark.unit

API_KEY = "unit-scheduler-key-" + "k" * 32
APPOINTMENT_ID = "apt_01JABCDEFGHJKMNPQRSTVWXYZ0"
REQUEST = AppointmentRequest(
    patient_ref="pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
    exams=(("LAB-10001", "Hemograma completo"), ("LAB-10043", "TSH")),
    source_document_sha256="ab" * 32,
    requested_by="exam_scheduling_agent",
)
KEY = "c" * 64
RUN_ID = "2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11"

type Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, requests: list[httpx.Request] | None = None) -> SchedulerClient:
    def record(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return handler(request)

    return SchedulerClient(
        "http://api.local:8000/", API_KEY, 10.0, transport=httpx.MockTransport(record)
    )


async def _create(client: SchedulerClient) -> AppointmentResult:
    return await client.create_appointment(REQUEST, idempotency_key=KEY, request_id=RUN_ID)


def _appointment(status: int, headers: dict[str, str] | None = None) -> Handler:
    return lambda request: httpx.Response(
        status, json={"appointment_id": APPOINTMENT_ID}, headers=headers
    )


async def test_created_sends_headers_and_body() -> None:
    requests: list[httpx.Request] = []

    result = await _create(_client(_appointment(201), requests))

    assert result == AppointmentResult("scheduled", APPOINTMENT_ID)
    [request] = requests
    assert (request.method, str(request.url)) == ("POST", "http://api.local:8000/v1/appointments")
    assert request.headers["X-API-Key"] == API_KEY
    assert request.headers["Idempotency-Key"] == KEY
    assert request.headers["X-Request-ID"] == RUN_ID
    assert json.loads(request.content) == {
        "patient_ref": "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
        "exams": [
            {"code": "LAB-10001", "name": "Hemograma completo"},
            {"code": "LAB-10043", "name": "TSH"},
        ],
        "source_document_sha256": "ab" * 32,
        "requested_by": "exam_scheduling_agent",
    }


async def test_replay_needs_the_replayed_header() -> None:
    replayed = await _create(_client(_appointment(200, {"Idempotent-Replayed": "true"})))

    assert replayed == AppointmentResult("replayed", APPOINTMENT_ID)
    with pytest.raises(PermanentError, match="SCHEDULER_UNEXPECTED_RESPONSE"):
        await _create(_client(_appointment(200)))


@pytest.mark.parametrize(
    ("status", "title"),
    [
        (400, "Invalid Idempotency-Key"),
        (401, "Unauthorized"),
        (409, "Idempotency key reuse"),
        (422, "Unknown exam code"),
    ],
)
async def test_client_errors_are_rejections_with_the_problem_title(status: int, title: str) -> None:
    requests: list[httpx.Request] = []

    def problem(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"type": "x", "title": title, "status": status},
            headers={"content-type": "application/problem+json"},
        )

    with pytest.raises(SchedulerRejected) as caught:
        await _create(_client(problem, requests))

    assert (caught.value.status, caught.value.title) == (status, title)
    assert len(requests) == 1


async def test_rejection_without_a_problem_body_uses_the_reason_phrase() -> None:
    with pytest.raises(SchedulerRejected) as caught:
        await _create(_client(lambda r: httpx.Response(422, text="<html>")))

    assert caught.value.title == "Unprocessable Entity"


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
async def test_server_errors_are_transient_and_not_retried(status: int) -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(TransientError):
        await _create(_client(lambda r: httpx.Response(status), requests))

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (httpx.ReadTimeout("slow"), "SCHEDULER_TIMEOUT"),
        (httpx.ConnectTimeout("slow"), "SCHEDULER_TIMEOUT"),
        (httpx.ConnectError("refused"), "SCHEDULER_UNAVAILABLE"),
        (httpx.RemoteProtocolError("bad"), "SCHEDULER_UNAVAILABLE"),
    ],
)
async def test_transport_failures_are_transient(exc: httpx.HTTPError, code: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(TransientError) as caught:
        await _create(_client(fail))

    assert caught.value.code == code
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(201, json={"appointment_id": "apt_x"}),
        httpx.Response(201, json=["apt"]),
        httpx.Response(201, text="not json"),
        httpx.Response(302, headers={"Location": "http://elsewhere/"}),
        httpx.Response(204),
    ],
)
async def test_unexpected_responses_are_permanent(response: httpx.Response) -> None:
    with pytest.raises(PermanentError) as caught:
        await _create(_client(lambda r: response))

    assert caught.value.code == "SCHEDULER_UNEXPECTED_RESPONSE"


async def test_client_is_reused_and_closed() -> None:
    client = _client(_appointment(201))
    await _create(client)
    first = client._client

    await _create(client)
    assert client._client is first
    await client.aclose()
    assert client._client is None
    await client.aclose()


def test_repr_hides_the_api_key() -> None:
    assert API_KEY not in repr(_client(_appointment(201)))
