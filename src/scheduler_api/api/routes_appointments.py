"""/v1/appointments: thin adapters between HTTP and `AppointmentService`."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Response, status

from scheduler_api.api import docs
from scheduler_api.api.deps import (
    AppointmentServiceDep,
    IdempotencyKey,
    document_request_id,
    require_api_key,
)
from scheduler_api.schemas import APPOINTMENT_ID_PATTERN, AppointmentCreate, AppointmentOut

router = APIRouter(
    prefix="/v1",
    tags=["appointments"],
    dependencies=[Depends(require_api_key), Depends(document_request_id)],
)

_APPOINTMENT_CONTENT = {"application/json": {"example": docs.APPOINTMENT_EXAMPLE}}


@router.post(
    "/appointments",
    status_code=status.HTTP_201_CREATED,
    summary="Request exam scheduling",
    description=(
        "Records a scheduling request with status `requested`. Exam codes are checked "
        "against the catalog. Idempotent per `Idempotency-Key`: the same key and body "
        "return the original appointment with `200`; the same key with a different "
        "body is a `409`."
    ),
    response_description="Appointment created.",
    responses={
        201: {
            "content": _APPOINTMENT_CONTENT,
            "headers": {
                "Location": {
                    "description": "Path of the new appointment.",
                    "schema": {"type": "string"},
                },
                "X-Request-ID": docs.REQUEST_ID_HEADER_DOC,
            },
        },
        200: {
            "model": AppointmentOut,
            "description": "Replay: same key and body as an earlier request; nothing new created.",
            "content": _APPOINTMENT_CONTENT,
            "headers": {
                "Idempotent-Replayed": {
                    "description": "Always `true` on a replay.",
                    "schema": {"type": "string", "enum": ["true"]},
                },
                "Location": {
                    "description": "Path of the original appointment.",
                    "schema": {"type": "string"},
                },
                "X-Request-ID": docs.REQUEST_ID_HEADER_DOC,
            },
        },
        **docs.problem_responses(
            {
                400: ["invalid_idempotency_key"],
                401: ["unauthorized"],
                409: ["idempotency_key_reuse"],
                422: ["unknown_exam_code", "exam_name_mismatch", "validation_error"],
                500: ["internal_error"],
            }
        ),
    },
)
def create_appointment(
    body: Annotated[AppointmentCreate, Body(openapi_examples=docs.CREATE_EXAMPLES)],
    response: Response,
    service: AppointmentServiceDep,
    idempotency_key: IdempotencyKey,
) -> AppointmentOut:
    appointment, replayed = service.create(body.to_domain(), idempotency_key)
    response.headers["Location"] = f"/v1/appointments/{appointment.id}"
    if replayed:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replayed"] = "true"
    return AppointmentOut.from_domain(appointment)


@router.get(
    "/appointments/{appointment_id}",
    summary="Get a scheduling request",
    response_description="The appointment.",
    responses={
        200: {"content": _APPOINTMENT_CONTENT},
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                404: ["appointment_not_found"],
                422: ["validation_error"],
                500: ["internal_error"],
            }
        ),
    },
)
def get_appointment(
    appointment_id: Annotated[
        str,
        Path(
            pattern=APPOINTMENT_ID_PATTERN,
            description="Id returned on creation (`Location` header).",
            examples=[docs.APPOINTMENT_ID],
        ),
    ],
    service: AppointmentServiceDep,
) -> AppointmentOut:
    return AppointmentOut.from_domain(service.get(appointment_id))
