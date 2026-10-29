from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol

import httpx
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_runtime.errors import AgentRuntimeError, PermanentError, TransientError
from lab_common import telemetry as lab_telemetry

APPOINTMENTS_PATH = "/v1/appointments"
API_KEY_HEADER = "X-API-Key"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
REQUEST_ID_HEADER = "X-Request-ID"
REPLAYED_HEADER = "Idempotent-Replayed"
CONNECT_TIMEOUT_S = 2.0
APPOINTMENT_ID_PATTERN = r"^apt_[0-9A-HJKMNP-TV-Z]{26}$"
_MAX_TITLE_CHARS = 200

tracer = lab_telemetry.tracer(__name__)


# Pseudonymous patient_ref, catalog codes and names, a hash and an app name: no personal data.
@dataclass(frozen=True, slots=True)
class AppointmentRequest:
    patient_ref: str
    exams: tuple[tuple[str, str], ...]
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


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _ExamOut(_Contract):
    code: str
    name: str


class AppointmentOut(_Contract):
    """The API's success body; tests/contract checks it against the OpenAPI schema."""

    appointment_id: Annotated[str, Field(pattern=APPOINTMENT_ID_PATTERN)]
    status: Literal["requested"]
    patient_ref: str
    exams: list[_ExamOut]
    source_document_sha256: str
    requested_by: str
    created_at: datetime


class SchedulerRejected(PermanentError):
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

    # Never retries: the node does, safely, since every attempt sends the same Idempotency-Key.
    async def create_appointment(
        self, request: AppointmentRequest, *, idempotency_key: str, request_id: str
    ) -> AppointmentResult:
        with tracer.start_as_current_span(
            f"POST {APPOINTMENTS_PATH}",
            kind=SpanKind.CLIENT,
            attributes={"http.request.method": "POST", "http.route": APPOINTMENTS_PATH},
            record_exception=False,
        ) as span:
            headers = lab_telemetry.inject_headers(
                {IDEMPOTENCY_KEY_HEADER: idempotency_key, REQUEST_ID_HEADER: request_id}
            )
            lab_telemetry.annotate(exam_count=len(request.exams))
            try:
                response = await self._post(request, headers)
                span.set_attribute("http.response.status_code", response.status_code)
                return _appointment_result(response, request)
            except AgentRuntimeError as error:
                lab_telemetry.fail(error.code, span)
                raise

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def _post(self, request: AppointmentRequest, headers: dict[str, str]) -> httpx.Response:
        try:
            return await self._http_client().post(
                APPOINTMENTS_PATH, json=request.payload(), headers=headers
            )
        except httpx.TimeoutException:
            raise TransientError(
                "SCHEDULER_TIMEOUT", "scheduler API did not answer in time"
            ) from None
        except httpx.HTTPError:
            raise TransientError("SCHEDULER_UNAVAILABLE", "scheduler API is unreachable") from None

    def _http_client(self) -> httpx.AsyncClient:
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


def _appointment_result(response: httpx.Response, request: AppointmentRequest) -> AppointmentResult:
    status = response.status_code
    if status == httpx.codes.CREATED:
        return AppointmentResult("scheduled", _confirmed_id(response, request))

    if status == httpx.codes.OK and response.headers.get(REPLAYED_HEADER) == "true":
        return AppointmentResult("replayed", _confirmed_id(response, request))

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


# Audit F04: an API regression must not show an appointment that does not match the order.
def _confirmed_id(response: httpx.Response, request: AppointmentRequest) -> str:
    try:
        body = AppointmentOut.model_validate(_json_object(response))
    except ValidationError:
        # Not chained: the error text would quote the body.
        raise PermanentError(
            "SCHEDULER_UNEXPECTED_RESPONSE", "response breaks the appointment contract"
        ) from None

    expected = (
        request.patient_ref,
        request.source_document_sha256,
        request.requested_by,
        sorted(code for code, _ in request.exams),
    )
    answered = (
        body.patient_ref,
        body.source_document_sha256,
        body.requested_by,
        sorted(exam.code for exam in body.exams),
    )
    if answered != expected:
        raise PermanentError("SCHEDULER_UNEXPECTED_RESPONSE", "response does not match the request")

    return body.appointment_id


def _problem_title(response: httpx.Response) -> str:
    body = _json_object(response)
    title = body.get("title") if body is not None else None
    if isinstance(title, str) and title.strip():
        return title.strip()[:_MAX_TITLE_CHARS]

    return response.reason_phrase or "error"
