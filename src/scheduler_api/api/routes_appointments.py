"""/v1/appointments: thin adapters between HTTP and `AppointmentService`."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Response, status

from scheduler_api.api.deps import AppointmentServiceDep, require_api_key
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
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> AppointmentOut:
    appointment = service.create(body.to_domain())
    response.headers["Location"] = f"/v1/appointments/{appointment.id}"
    return AppointmentOut.from_domain(appointment)


@router.get("/appointments/{appointment_id}", summary="Get a scheduling request")
def get_appointment(
    appointment_id: Annotated[str, Path(pattern=APPOINTMENT_ID_PATTERN)],
    service: AppointmentServiceDep,
) -> AppointmentOut:
    return AppointmentOut.from_domain(service.get(appointment_id))
