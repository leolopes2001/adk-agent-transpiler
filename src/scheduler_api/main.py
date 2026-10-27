"""Application factory.

`create_app` builds a new app per call and keeps every dependency on
`app.state`, so tests can create isolated apps and there is no module-level
state. Run with `uvicorn scheduler_api.main:create_app --factory`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scheduler_api.api import routes_health
from scheduler_api.api.middleware import RequestContextMiddleware
from scheduler_api.log import configure_logging, get_logger
from scheduler_api.settings import Settings

API_VERSION = "1.0.0"

_log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The database check is a stub until the persistence layer exists.
    app.state.readiness_checks = {"database": lambda: True}
    _log.info("startup")
    yield
    _log.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]  # fields come from the environment
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Lab Exam Scheduler API",
        version=API_VERSION,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.include_router(routes_health.router)
    return app
