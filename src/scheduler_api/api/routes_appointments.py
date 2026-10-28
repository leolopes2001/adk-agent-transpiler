from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query, Response, status

from scheduler_api.api import docs
from scheduler_api.api.deps import (
    AppointmentServiceDep,
    IdempotencyKey,
    document_request_id,
    require_api_key,
)
from scheduler_api.api.middleware import REQUEST_ID_HEADER
from scheduler_api.schemas import (
    APPOINTMENT_ID_PATTERN,
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    PATIENT_REF_PATTERN,
    AppointmentCreate,
    AppointmentOut,
)

router = APIRouter(
    prefix="/v1",
    tags=["appointments"],
    dependencies=[Depends(require_api_key), Depends(document_request_id)],
)

_REPLAYED_HEADER = "Idempotent-Replayed"
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
                REQUEST_ID_HEADER: docs.REQUEST_ID_HEADER_DOC,
            },
        },
        200: {
            "model": AppointmentOut,
            "description": "Replay: same key and body as an earlier request; nothing new created.",
            "content": _APPOINTMENT_CONTENT,
            "headers": {
                _REPLAYED_HEADER: {
                    "description": "Always `true` on a replay.",
                    "schema": {"type": "string", "enum": ["true"]},
                },
                "Location": {
                    "description": "Path of the original appointment.",
                    "schema": {"type": "string"},
                },
                REQUEST_ID_HEADER: docs.REQUEST_ID_HEADER_DOC,
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
        response.headers[_REPLAYED_HEADER] = "true"

    return AppointmentOut.from_domain(appointment)


@router.get(
    "/appointments",
    summary="List scheduling requests",
    description="Newest first, optionally only those of one `patient_ref`.",
    response_description="The appointments.",
    responses={
        200: {"content": {"application/json": {"example": [docs.APPOINTMENT_EXAMPLE]}}},
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                422: ["validation_error"],
                500: ["internal_error"],
            }
        ),
    },
)
def list_appointments(
    service: AppointmentServiceDep,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_LIST_LIMIT, description=f"1 to {MAX_LIST_LIMIT} appointments."),
    ] = DEFAULT_LIST_LIMIT,
    patient_ref: Annotated[
        str | None,
        Query(
            pattern=PATIENT_REF_PATTERN,
            description="Only this patient's appointments.",
            examples=[docs.PATIENT_REF],
        ),
    ] = None,
) -> list[AppointmentOut]:
    return [
        AppointmentOut.from_domain(appointment)
        for appointment in service.list_recent(limit, patient_ref)
    ]


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
