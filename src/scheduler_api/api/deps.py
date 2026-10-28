import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from scheduler_api.api.docs import IDEMPOTENCY_KEY_EXAMPLE
from scheduler_api.api.errors import INVALID_API_KEY_DETAIL
from scheduler_api.api.middleware import REQUEST_ID_HEADER
from scheduler_api.domain.catalog import Catalog
from scheduler_api.domain.services import AppointmentService
from scheduler_api.log import get_logger
from scheduler_api.settings import Settings

API_KEY_HEADER = "X-API-Key"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9_\-:.]{8,128}$"

_log = get_logger(__name__)

# auto_error=False so a missing header goes through the same 401 path as a wrong one.
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Shared API key configured on the server (`SCHEDULER_API_KEY`).",
)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    expected = settings.scheduler_api_key.get_secret_value().encode()
    # Constant-time comparison; bytes so a non-ASCII header cannot raise TypeError.
    if api_key is None or not hmac.compare_digest(api_key.encode(), expected):
        _log.warning("auth_failed", reason="missing" if api_key is None else "invalid")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_API_KEY_DETAIL)


def get_catalog(request: Request) -> Catalog:
    catalog: Catalog = request.app.state.catalog
    return catalog


def get_appointment_service(request: Request) -> AppointmentService:
    service: AppointmentService = request.app.state.appointment_service
    return service


CatalogDep = Annotated[Catalog, Depends(get_catalog)]
AppointmentServiceDep = Annotated[AppointmentService, Depends(get_appointment_service)]

IdempotencyKey = Annotated[
    str,
    Header(
        alias=IDEMPOTENCY_KEY_HEADER,
        min_length=8,
        max_length=128,
        pattern=IDEMPOTENCY_KEY_PATTERN,
        description=(
            "Client-chosen key, 8 to 128 characters from `A-Z a-z 0-9 _ - : .`. "
            "Repeating a request with the same key and body returns the original "
            "appointment instead of creating another one."
        ),
        openapi_examples={
            "example": {
                "summary": "Key chosen by the client",
                "value": IDEMPOTENCY_KEY_EXAMPLE,
            }
        },
    ),
]


def document_request_id(
    x_request_id: Annotated[
        str | None,
        Header(
            alias=REQUEST_ID_HEADER,
            description=(
                "Optional correlation id, echoed in the response and in the logs. "
                "Values outside `[A-Za-z0-9_-:.]{1,128}` are replaced by a generated UUID."
            ),
        ),
    ] = None,
) -> None:
    """Only documents the header; `RequestContextMiddleware` does the work."""
