"""Application factory.

`create_app` builds a new app per call and keeps every dependency on
`app.state`, so tests can create isolated apps and there is no module-level
state. Run with `uvicorn scheduler_api.main:create_app --factory`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scheduler_api.api import routes_health
from scheduler_api.api.errors import install_error_handlers
from scheduler_api.api.middleware import RequestContextMiddleware
from scheduler_api.infra.db import make_engine, make_session_factory, ping
from scheduler_api.log import configure_logging, get_logger
from scheduler_api.settings import Settings

API_VERSION = "1.0.0"

_log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # The engine connects lazily: the app starts even if Postgres is not up yet,
    # and /health/ready reports it until it is.
    engine = make_engine(settings.database_url.get_secret_value())
    app.state.session_factory = make_session_factory(engine)
    app.state.readiness_checks = {"database": lambda: ping(engine)}
    _log.info("startup")
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
    app.include_router(routes_health.router)
    return app
