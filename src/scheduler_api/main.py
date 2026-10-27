"""Application factory.

`create_app` builds a new app per call and keeps every dependency on
`app.state`, so tests can create isolated apps and there is no module-level
state. Run with `uvicorn scheduler_api.main:create_app --factory`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI

from scheduler_api.api import routes_appointments, routes_exams, routes_health
from scheduler_api.api.errors import install_error_handlers
from scheduler_api.api.middleware import RequestContextMiddleware
from scheduler_api.domain.catalog import Catalog
from scheduler_api.domain.services import AppointmentService
from scheduler_api.infra.db import make_engine, make_session_factory, ping
from scheduler_api.infra.repositories import SqlUnitOfWork
from scheduler_api.log import configure_logging, get_logger
from scheduler_api.settings import Settings

API_VERSION = "1.0.0"

_log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # The engine connects lazily: the app starts even if Postgres is not up yet,
    # and /health/ready reports it until it is.
    # A missing or malformed catalog is a deployment error: fail at startup.
    catalog = Catalog.load(settings.catalog_path)
    engine = make_engine(settings.database_url.get_secret_value())
    session_factory = make_session_factory(engine)
    app.state.catalog = catalog
    app.state.appointment_service = AppointmentService(
        lambda: SqlUnitOfWork(session_factory),
        catalog,
        idempotency_ttl=timedelta(hours=settings.idempotency_ttl_hours),
    )
    app.state.readiness_checks = {
        "database": lambda: ping(engine),
        "catalog": lambda: len(catalog) > 0,
    }
    _log.info("startup", catalog_version=catalog.version, catalog_exams=len(catalog))
    try:
        yield
    finally:
        engine.dispose()
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
    install_error_handlers(app)
    app.include_router(routes_appointments.router)
    app.include_router(routes_exams.router)
    app.include_router(routes_health.router)
    return app
