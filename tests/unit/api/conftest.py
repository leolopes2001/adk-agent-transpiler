import io
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from fastapi import FastAPI

from scheduler_api import log
from scheduler_api.main import create_app
from scheduler_api.settings import Settings

CATALOG = Path(__file__).parents[3] / "data" / "exams_catalog.json"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    if log._handler is not None:
        root.removeHandler(log._handler)
        log._handler = None
    root.setLevel(level)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture
def settings() -> Settings:
    # Unit tests never reach this database; port 1 makes any accidental use fail fast.
    return Settings(
        database_url="postgresql+psycopg://unit:unit@127.0.0.1:1/unit",
        scheduler_api_key="unit-test-key-0123456789abcdefghijklmnop",
        catalog_path=CATALOG,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def log_stream(app: FastAPI) -> io.StringIO:
    """Redirects the logging configured by `create_app` to a buffer."""
    stream = io.StringIO()
    log.configure_logging(stream=stream)
    return stream


@pytest.fixture
def log_records(log_stream: io.StringIO) -> Callable[[], list[dict[str, Any]]]:
    def read() -> list[dict[str, Any]]:
        return [json.loads(line) for line in log_stream.getvalue().splitlines()]

    return read
