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
    Appointment,
    NewAppointment,
)

router = APIRouter(
    prefix="/v1",
    tags=[docs.TAG_APPOINTMENTS],
    dependencies=[Depends(require_api_key), Depends(document_request_id)],
)

_REPLAYED_HEADER = "Idempotent-Replayed"
_APPOINTMENT_CONTENT = {"application/json": {"example": docs.APPOINTMENT_EXAMPLE}}
_REQUEST_ID_HEADERS = {REQUEST_ID_HEADER: docs.REQUEST_ID_HEADER_DOC}


@router.post(
    "/appointments",
    status_code=status.HTTP_201_CREATED,
    summary="Solicitar agendamento de exames",
    description=(
        "Registra uma solicitação com status `requested`. Os códigos são conferidos no "
        "catálogo. Idempotente por `Idempotency-Key`: a mesma chave com o mesmo corpo "
        "devolve a solicitação original com `200`; com outro corpo, `409`."
    ),
    response_description="Solicitação criada.",
    responses={
        201: {
            "content": _APPOINTMENT_CONTENT,
            "headers": {
                "Location": {
                    "description": "Path da nova solicitação.",
                    "schema": {"type": "string"},
                },
                **_REQUEST_ID_HEADERS,
            },
        },
        200: {
            "model": Appointment,
            "description": "Replay: mesma chave e mesmo corpo de antes; nada novo foi criado.",
            "content": _APPOINTMENT_CONTENT,
            "headers": {
                _REPLAYED_HEADER: {
                    "description": "Sempre `true` num replay.",
                    "schema": {"type": "string", "enum": ["true"]},
                },
                "Location": {
                    "description": "Path da solicitação original.",
                    "schema": {"type": "string"},
                },
                **_REQUEST_ID_HEADERS,
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
    body: Annotated[NewAppointment, Body(openapi_examples=docs.CREATE_EXAMPLES)],
    response: Response,
    service: AppointmentServiceDep,
    idempotency_key: IdempotencyKey,
) -> Appointment:
    appointment, replayed = service.create(body.to_domain(), idempotency_key)
    response.headers["Location"] = f"/v1/appointments/{appointment.id}"
    if replayed:
        response.status_code = status.HTTP_200_OK
        response.headers[_REPLAYED_HEADER] = "true"

    return Appointment.from_domain(appointment)


@router.get(
    "/appointments",
    summary="Listar solicitações",
    description=(
        "Devolve as solicitações da mais recente para a mais antiga, opcionalmente só as "
        "de um `patient_ref`."
    ),
    response_description="As solicitações.",
    responses={
        200: {
            "content": {"application/json": {"example": [docs.APPOINTMENT_EXAMPLE]}},
            "headers": _REQUEST_ID_HEADERS,
        },
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                422: ["invalid_limit", "invalid_patient_ref"],
                500: ["internal_error"],
            }
        ),
    },
)
def list_appointments(
    service: AppointmentServiceDep,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_LIST_LIMIT,
            description=f"Quantidade máxima de solicitações, de 1 a {MAX_LIST_LIMIT}.",
        ),
    ] = DEFAULT_LIST_LIMIT,
    patient_ref: Annotated[
        str | None,
        Query(
            pattern=PATIENT_REF_PATTERN,
            description="Só as solicitações deste pseudônimo (`pt_` + 26 caracteres A-Z, 2-7).",
            openapi_examples={
                "patient_ref": {"summary": "Pseudônimo válido", "value": docs.PATIENT_REF}
            },
        ),
    ] = None,
) -> list[Appointment]:
    return [
        Appointment.from_domain(appointment)
        for appointment in service.list_recent(limit, patient_ref)
    ]


@router.get(
    "/appointments/{appointment_id}",
    summary="Consultar uma solicitação",
    description="Devolve a solicitação pelo id que o `POST` devolveu e o agente imprime.",
    response_description="A solicitação.",
    responses={
        200: {"content": _APPOINTMENT_CONTENT, "headers": _REQUEST_ID_HEADERS},
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                404: ["appointment_not_found"],
                422: ["invalid_appointment_id"],
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
            description="Id da solicitação (`apt_` + ULID), como no header `Location` do `201`.",
            openapi_examples={
                "appointment_id": {"summary": "Id no formato", "value": docs.APPOINTMENT_ID}
            },
        ),
    ],
    service: AppointmentServiceDep,
) -> Appointment:
    return Appointment.from_domain(service.get(appointment_id))
