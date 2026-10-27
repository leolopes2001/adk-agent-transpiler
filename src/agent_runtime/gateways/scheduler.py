"""HTTP client for the scheduler API (`POST /v1/appointments`).

The client never retries: a 5xx, a timeout or a connection failure raises
`TransientError` and the schedule node's RetryConfig decides. That is safe
because every request carries the same `Idempotency-Key` (ADR-0006). A 4xx is
final and raises `SchedulerRejected` with the problem title, which the node
turns into a `failed` outcome.

The request body holds only the pseudonymous `patient_ref`, catalog codes and
names, the document hash and the requester id; no personal data exists here.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from agent_runtime.errors import PermanentError, TransientError

APPOINTMENTS_PATH = "/v1/appointments"
API_KEY_HEADER = "X-API-Key"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
REQUEST_ID_HEADER = "X-Request-ID"
REPLAYED_HEADER = "Idempotent-Replayed"
CONNECT_TIMEOUT_S = 2.0
_APPOINTMENT_ID = re.compile(r"^apt_[0-9A-HJKMNP-TV-Z]{26}$")
_MAX_TITLE_CHARS = 200


@dataclass(frozen=True, slots=True)
class AppointmentRequest:
    patient_ref: str
    exams: tuple[tuple[str, str], ...]
    """(code, catalog name) pairs."""
    source_document_sha256: str
    requested_by: str

    def payload(self) -> dict[str, Any]:
        return {
            "patient_ref": self.patient_ref,
            "exams": [{"code": code, "name": name} for code, name in self.exams],
            "source_document_sha256": self.source_document_sha256,
            "requested_by": self.requested_by,
        }


@dataclass(frozen=True, slots=True)
class AppointmentResult:
    status: Literal["scheduled", "replayed"]
    appointment_id: str


class SchedulerRejected(PermanentError):
    """The API refused the request (4xx)."""

    def __init__(self, status: int, title: str) -> None:
        super().__init__("SCHEDULER_REJECTED", f"HTTP {status}: {title}")
        self.status = status
        self.title = title


class SchedulerGateway(Protocol):
    async def create_appointment(
        self, request: AppointmentRequest, *, idempotency_key: str, request_id: str
    ) -> AppointmentResult: ...

    async def aclose(self) -> None: ...


class SchedulerClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = httpx.Timeout(timeout_s, connect=min(CONNECT_TIMEOUT_S, timeout_s))
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def __repr__(self) -> str:
        return f"SchedulerClient(base_url={self._base_url!r})"

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Not following redirects keeps the API key on the configured host.
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
                follow_redirects=False,
                headers={API_KEY_HEADER: self._api_key},
            )
        return self._client

    async def create_appointment(
        self, request: AppointmentRequest, *, idempotency_key: str, request_id: str
    ) -> AppointmentResult:
        headers = {IDEMPOTENCY_KEY_HEADER: idempotency_key, REQUEST_ID_HEADER: request_id}
        try:
            response = await self._http().post(
                APPOINTMENTS_PATH, json=request.payload(), headers=headers
            )
        except httpx.TimeoutException:
            raise TransientError(
                "SCHEDULER_TIMEOUT", "scheduler API did not answer in time"
            ) from None
        except httpx.HTTPError:
            raise TransientError("SCHEDULER_UNAVAILABLE", "scheduler API is unreachable") from None
        return _result(response)

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def _result(response: httpx.Response) -> AppointmentResult:
    status = response.status_code
    if status == httpx.codes.CREATED:
        return AppointmentResult("scheduled", _appointment_id(response))
    if status == httpx.codes.OK and response.headers.get(REPLAYED_HEADER) == "true":
        return AppointmentResult("replayed", _appointment_id(response))
    if status >= httpx.codes.INTERNAL_SERVER_ERROR or status in (
        httpx.codes.REQUEST_TIMEOUT,
        httpx.codes.TOO_MANY_REQUESTS,
    ):
        raise TransientError("SCHEDULER_UNAVAILABLE", f"HTTP {status}")
    if httpx.codes.BAD_REQUEST <= status < httpx.codes.INTERNAL_SERVER_ERROR:
        raise SchedulerRejected(status, _problem_title(response))
    raise PermanentError("SCHEDULER_UNEXPECTED_RESPONSE", f"HTTP {status}")


def _json_object(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, Mapping) else None


def _appointment_id(response: httpx.Response) -> str:
    body = _json_object(response)
    value = body.get("appointment_id") if body is not None else None
    if not isinstance(value, str) or not _APPOINTMENT_ID.match(value):
        raise PermanentError("SCHEDULER_UNEXPECTED_RESPONSE", "response has no appointment_id")
    return value


def _problem_title(response: httpx.Response) -> str:
    body = _json_object(response)
    title = body.get("title") if body is not None else None
    if isinstance(title, str) and title.strip():
        return title.strip()[:_MAX_TITLE_CHARS]
    return response.reason_phrase or "error"
