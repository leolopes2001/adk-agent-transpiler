"""/v1/appointments: thin adapters between HTTP and `AppointmentService`."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from scheduler_api.api.deps import AppointmentServiceDep, IdempotencyKey, require_api_key
from scheduler_api.schemas import APPOINTMENT_ID_PATTERN, AppointmentCreate, AppointmentOut

router = APIRouter(prefix="/v1", tags=["appointments"], dependencies=[Depends(require_api_key)])


@router.post(
    "/appointments",
    status_code=status.HTTP_201_CREATED,
    summary="Request exam scheduling",
)
def create_appointment(
    body: AppointmentCreate,
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


@router.get("/appointments/{appointment_id}", summary="Get a scheduling request")
def get_appointment(
    appointment_id: Annotated[str, Path(pattern=APPOINTMENT_ID_PATTERN)],
    service: AppointmentServiceDep,
) -> AppointmentOut:
    return AppointmentOut.from_domain(service.get(appointment_id))
