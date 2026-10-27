"""RFC 9457 problem details for every error response.

All errors leave the API as `application/problem+json` with a stable `type`
URI per error code, the request id and, for validation errors, one entry per
failing field. Client input is never echoed back: validation errors carry the
location, the error code and pydantic's message, not the offending value.
"""

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from scheduler_api.domain.models import AppointmentNotFoundError, CatalogValidationError

PROBLEM_MEDIA_TYPE = "application/problem+json"
TYPE_BASE = "https://errors.lab-agent.example/"

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
    loc: list[str | int] = Field(description="Path to the offending field, e.g. body.exams.1.code.")
    code: str = Field(description="Machine-readable error code.")
    msg: str = Field(description="Human-readable message; never contains the submitted value.")


class Problem(BaseModel):
    """Error body (RFC 9457)."""

    type: str = Field(description="URI identifying the error code.")
    title: str = Field(description="Short summary of the error code.")
    status: int = Field(description="HTTP status code.")
    detail: str | None = Field(default=None, description="Explanation of this occurrence.")
    instance: str | None = Field(default=None, description="Path of the request.")
    request_id: str | None = Field(default=None, description="Same value as `X-Request-ID`.")
    errors: list[FieldError] | None = Field(
        default=None, description="Per-field errors (validation failures only)."
    )


class ApiError(Exception):
    """Raised by the HTTP layer to answer with a specific problem."""

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
    return problem_response(
        scope,
        status=500,
        code="internal_error",
        title="Internal server error",
        detail="The request could not be completed. Quote the request_id when reporting it.",
    )


def field_errors(raw: Sequence[Any]) -> list[FieldError]:
    """Converts pydantic error dicts, dropping `input` and `ctx` (they may hold client data)."""
    return [
        FieldError(loc=list(error.get("loc", ())), code=error["type"], msg=error["msg"])
        for error in raw
    ]


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
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


async def _http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    status = exc.status_code
    detail = (
        exc.detail
        if isinstance(exc.detail, str) and status < HTTPStatus.INTERNAL_SERVER_ERROR
        else None
    )
    if detail == HTTPStatus(status).phrase:
        detail = None
    return problem_response(
        request.scope,
        status=status,
        code=_CODE_BY_STATUS.get(status, "http_error"),
        title=HTTPStatus(status).phrase,
        detail=detail,
        headers=exc.headers,
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    errors = field_errors(exc.errors())
    return problem_response(
        request.scope,
        status=422,
        code="validation_error",
        title="Request validation failed",
        detail=f"{len(errors)} validation error(s).",
        errors=errors,
    )


async def _catalog_validation(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, CatalogValidationError)  # noqa: S101 - registered for it only
    unknown = exc.count("unknown_exam_code")
    if unknown:
        code, title = "unknown_exam_code", "Unknown exam code"
        detail = f"{unknown} exam code(s) do not exist in the catalog."
    else:
        code, title = "exam_name_mismatch", "Exam name mismatch"
        detail = f"{exc.count('exam_name_mismatch')} exam name(s) do not match the catalog."
    errors = [
        FieldError(
            loc=["body", "exams", issue.index, issue.field], code=issue.kind, msg=issue.message
        )
        for issue in exc.issues
    ]
    return problem_response(
        request.scope, status=422, code=code, title=title, detail=detail, errors=errors
    )


async def _appointment_not_found(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppointmentNotFoundError)  # noqa: S101 - registered for it only
    return problem_response(
        request.scope,
        status=404,
        code="appointment_not_found",
        title="Appointment not found",
        detail=f"{exc.appointment_id} does not exist.",
    )


def install_error_handlers(app: FastAPI) -> None:
    """Unexpected exceptions are handled by `RequestContextMiddleware` (it owns the request id)."""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(CatalogValidationError, _catalog_validation)
    app.add_exception_handler(AppointmentNotFoundError, _appointment_not_found)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
