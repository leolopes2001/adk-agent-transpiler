"""Liveness and readiness probes (no authentication)."""

from collections.abc import Callable, Mapping
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from scheduler_api.log import get_logger

router = APIRouter(prefix="/health", tags=["health"])

_log = get_logger(__name__)

ReadinessChecks = Mapping[str, Callable[[], bool]]


class HealthStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, Literal["ok", "error"]] = {}


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
    results: dict[str, Literal["ok", "error"]] = {}
    for name, check in checks.items():
        try:
            passed = check()
        except Exception:  # noqa: BLE001 - any failure means "not ready"; details stay in the log
            _log.warning("readiness_check_failed", check=name, exc_info=True)
            passed = False
        results[name] = "ok" if passed else "error"

    if all(result == "ok" for result in results.values()):
        return HealthStatus(status="ok", checks=results)
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(status="unavailable", checks=results)
