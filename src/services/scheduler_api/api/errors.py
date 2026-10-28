from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from scheduler_api.domain.models import (
    AppointmentNotFoundError,
    CatalogValidationError,
    IdempotencyKeyReuseError,
)

IDEMPOTENCY_KEY_LOC = ("header", "idempotency-key")

PROBLEM_MEDIA_TYPE = "application/problem+json"
TYPE_BASE = "https://errors.lab-agent.example/"

PROBLEM_TITLES = {
    "invalid_idempotency_key": "Invalid Idempotency-Key",
    "idempotency_key_reuse": "Idempotency key reuse",
    "unknown_exam_code": "Unknown exam code",
    "exam_name_mismatch": "Exam name mismatch",
    "validation_error": "Request validation failed",
    "appointment_not_found": "Appointment not found",
    "exam_not_found": "Exam not found",
    "internal_error": "Internal server error",
}
INVALID_API_KEY_DETAIL = "Missing or invalid API key."
INVALID_IDEMPOTENCY_KEY_DETAIL = (
    "Idempotency-Key is required: 8 to 128 characters from A-Z a-z 0-9 _ - : ."
)
IDEMPOTENCY_KEY_REUSE_DETAIL = (
    "This Idempotency-Key was already used with a different request body."
)
INTERNAL_ERROR_DETAIL = (
    "The request could not be completed. Quote the request_id when reporting it."
)

_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    409: "conflict",
    413: "content_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}


class FieldError(BaseModel):
    loc: list[str | int] = Field(description="Caminho do campo com erro, ex.: body.exams.1.code.")
    code: str = Field(description="Código do erro, para máquinas.")
    msg: str = Field(description="Mensagem legível; nunca contém o valor enviado.")


class Problem(BaseModel):
    """Corpo de erro (RFC 9457)."""

    type: str = Field(description="URI que identifica o código do erro.")
    title: str = Field(description="Resumo do código do erro.")
    status: int = Field(description="Status HTTP.")
    detail: str | None = Field(default=None, description="Explicação desta ocorrência.")
    instance: str | None = Field(default=None, description="Path da requisição.")
    request_id: str | None = Field(default=None, description="Mesmo valor do `X-Request-ID`.")
    errors: list[FieldError] | None = Field(
        default=None, description="Erros por campo (só em falhas de validação)."
    )


class ApiError(Exception):
    def __init__(  # noqa: PLR0913 - one argument per problem field
        self,
        status: int,
        code: str,
        title: str,
        detail: str | None = None,
        *,
        errors: Sequence[FieldError] = (),
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail or title)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.errors = list(errors)
        self.headers = dict(headers or {})


def problem_type(code: str) -> str:
    return TYPE_BASE + code


def problem_response(  # noqa: PLR0913 - one argument per problem field
    scope: Scope,
    *,
    status: int,
    code: str,
    title: str,
    detail: str | None = None,
    errors: Sequence[FieldError] = (),
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    problem = Problem(
        type=problem_type(code),
        title=title,
        status=status,
        detail=detail,
        instance=scope.get("path"),
        request_id=scope.get("state", {}).get("request_id"),
        errors=list(errors) or None,
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=dict(headers or {}),
        media_type=PROBLEM_MEDIA_TYPE,
    )


def internal_error_response(scope: Scope) -> JSONResponse:
    # Deliberately generic: the cause is in the log, keyed by request_id.
    return _known_problem(scope, 500, "internal_error", INTERNAL_ERROR_DETAIL)


def field_errors(raw: Sequence[Any]) -> list[FieldError]:
    """Drops pydantic's `input` and `ctx`: they may hold client data."""
    return [
        FieldError(loc=list(error.get("loc", ())), code=error["type"], msg=error["msg"])
        for error in raw
    ]


def install_error_handlers(app: FastAPI) -> None:
    """Unexpected exceptions are handled by `RequestContextMiddleware` (it owns the request id)."""
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(CatalogValidationError, _handle_catalog_validation_error)
    app.add_exception_handler(AppointmentNotFoundError, _handle_appointment_not_found)
    app.add_exception_handler(IdempotencyKeyReuseError, _handle_idempotency_key_reuse)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_request_validation_error)


def _known_problem(
    scope: Scope,
    status: int,
    code: str,
    detail: str,
    errors: Sequence[FieldError] = (),
) -> JSONResponse:
    return problem_response(
        scope,
        status=status,
        code=code,
        title=PROBLEM_TITLES[code],
        detail=detail,
        errors=errors,
    )


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)  # noqa: S101 - registered for ApiError only
    return problem_response(
        request.scope,
        status=exc.status,
        code=exc.code,
        title=exc.title,
        detail=exc.detail,
        errors=exc.errors,
        headers=exc.headers,
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    title = HTTPStatus(exc.status_code).phrase
    return problem_response(
        request.scope,
        status=exc.status_code,
        code=_CODE_BY_STATUS.get(exc.status_code, "http_error"),
        title=title,
        detail=_client_safe_detail(exc, title),
        headers=exc.headers,
    )


async def _handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    errors = field_errors(exc.errors())
    key_errors = [error for error in errors if _is_idempotency_key_error(error)]
    # A bad key is a protocol error (400), reported alone: the body is not worth validating.
    if key_errors:
        return _known_problem(
            request.scope,
            400,
            "invalid_idempotency_key",
            INVALID_IDEMPOTENCY_KEY_DETAIL,
            key_errors,
        )

    return _known_problem(
        request.scope,
        422,
        "validation_error",
        f"{len(errors)} validation error(s).",
        errors,
    )


async def _handle_catalog_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, CatalogValidationError)  # noqa: S101 - registered for it only
    unknown_codes = exc.count("unknown_exam_code")
    if unknown_codes:
        code = "unknown_exam_code"
        detail = f"{unknown_codes} exam code(s) do not exist in the catalog."
    else:
        code = "exam_name_mismatch"
        detail = f"{exc.count('exam_name_mismatch')} exam name(s) do not match the catalog."

    errors = [
        FieldError(
            loc=["body", "exams", issue.index, issue.field], code=issue.kind, msg=issue.message
        )
        for issue in exc.issues
    ]
    return _known_problem(request.scope, 422, code, detail, errors)


async def _handle_appointment_not_found(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppointmentNotFoundError)  # noqa: S101 - registered for it only
    return _known_problem(
        request.scope,
        404,
        "appointment_not_found",
        f"{exc.appointment_id} does not exist.",
    )


async def _handle_idempotency_key_reuse(request: Request, exc: Exception) -> JSONResponse:
    return _known_problem(
        request.scope,
        409,
        "idempotency_key_reuse",
        IDEMPOTENCY_KEY_REUSE_DETAIL,
    )


def _client_safe_detail(exc: StarletteHTTPException, title: str) -> str | None:
    if exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR or exc.detail == title:
        return None

    return exc.detail if isinstance(exc.detail, str) else None


def _is_idempotency_key_error(error: FieldError) -> bool:
    location = tuple(str(part).lower() for part in error.loc[:2])
    return location == IDEMPOTENCY_KEY_LOC
