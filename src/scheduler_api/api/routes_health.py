from collections.abc import Callable, Mapping
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from scheduler_api.log import get_logger

router = APIRouter(prefix="/health", tags=["health"])

_log = get_logger(__name__)

CheckResult = Literal["ok", "error"]
ReadinessCheck = Callable[[], bool]
ReadinessChecks = Mapping[str, ReadinessCheck]


class HealthStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, CheckResult] = {}


@router.get("/live", summary="Process is alive")
def live() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get(
    "/ready",
    summary="Dependencies are reachable",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthStatus}},
)
def ready(request: Request, response: Response) -> HealthStatus:
    checks: ReadinessChecks = request.app.state.readiness_checks
    results = {name: _run_check(name, check) for name, check in checks.items()}
    if all(result == "ok" for result in results.values()):
        return HealthStatus(status="ok", checks=results)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(status="unavailable", checks=results)


def _run_check(name: str, check: ReadinessCheck) -> CheckResult:
    try:
        passed = check()
    except Exception:  # noqa: BLE001 - any failure means "not ready"; details stay in the log
        _log.warning("readiness_check_failed", check=name, exc_info=True)
        return "error"

    return "ok" if passed else "error"
