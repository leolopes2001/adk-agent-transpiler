"""The whole runtime stack in one test process, for integration and security tests.

Postgres runs in a container (testcontainers); the scheduler API, the OCR
server (real Tesseract) and the RAG server run in background threads on
ephemeral ports, built as their `__main__` builds them. The agent package is
generated from the versioned spec by the transpiler, and `lab-agent` runs it
in process with the FakeLlm. `tests/` is on sys.path through its conftest,
so test directories import this module by name.
"""

import io
import logging
import shutil
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
import structlog
import uvicorn
from alembic import command
from alembic.config import Config
from google.adk.models import LlmRequest
from google.adk.sessions import InMemorySessionService, Session
from sqlalchemy import create_engine, text
from starlette.types import ASGIApp
from testcontainers.community.postgres import PostgresContainer

from agent_runtime import cli
from agent_runtime.models import FakeLlm
from agent_runtime.settings import AgentSettings
from lab_common import logging as lab_logging
from lab_common.mcp_server import LibraryLogFilter
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.server import build_app as build_ocr_app
from ocr_mcp.settings import OcrSettings
from rag_mcp.catalog import load_catalog
from rag_mcp.server import build_app as build_rag_app
from rag_mcp.settings import RagSettings
from scheduler_api.main import create_app
from scheduler_api.settings import Settings
from transpiler.pipeline import generate

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "specs" / "exam_scheduling_agent.json"
CATALOG = REPO_ROOT / "data" / "exams_catalog.json"
FAKE_LLM_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "fake_llm"
SAMPLES = ("pedido_medico_ficticio.png", "pedido_injecao.png", "pedido_ambiguo.png")
PACKAGE = "exam_scheduling_agent"
MCP_TOKEN = "stack-mcp-token-" + "x" * 32
HMAC_SECRET = "stack-hmac-secret-" + "y" * 32
API_KEY = "stack-api-key-" + "z" * 32
POSTGRES_IMAGE = "postgres:16-alpine"
SAMPLE_PII = (
    # Printed on the samples by scripts/make_sample_images.py (all fictional).
    "Maria",
    "MARIA",
    "Fictícia da Silva",
    "202.393.613-69",
    "20239361369",
    "14/03/1985",
    "(31) 90000-0000",
    "90000-0000",
    "0000 1111 2222 3333",
    "João Exemplo",
    "CRM-MG 000000",
    "Rua Imaginária",
    # tests/data/images/nomes_sem_capitalizacao.png (audit F01).
    "maria",
    "fictícia da silva",
    "joana",
    "FICCIONAL",
    "teixeira",
    "beatriz",
    "Inventada",
)
"""Personal data printed on the images the PII tests run."""


def serve(app: ASGIApp) -> Iterator[str]:
    """Run `app` until the generator is closed; yields its base URL."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        threading.Event().wait(0.05)
    assert server.started, "server did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextmanager
def _served(app: ASGIApp) -> Iterator[str]:
    generator = serve(app)
    try:
        yield next(generator)
    finally:
        generator.close()


def forget_generated_package() -> None:
    """Drop the generated package from the import system (another copy may come next)."""
    for name in [m for m in sys.modules if m.split(".")[0] == PACKAGE]:
        del sys.modules[name]


@dataclass(frozen=True)
class Stack:
    work: Path
    """INPUT_ROOT: holds the sample images and the generated package."""
    agent_dir: Path
    database_url: str
    env: dict[str, str]

    def count(self, table: str) -> int:
        engine = create_engine(self.database_url)
        try:
            with engine.connect() as connection:
                value = connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()  # noqa: S608 - fixed names
        finally:
            engine.dispose()
        return int(value)

    def dump_database(self) -> str:
        """Every row of every table, as text (for leak checks)."""
        engine = create_engine(self.database_url)
        rows: list[str] = []
        try:
            with engine.connect() as connection:
                for table in ("appointments", "appointment_exams", "idempotency_keys"):
                    result = connection.execute(text(f"SELECT * FROM {table}"))  # noqa: S608
                    rows.extend(repr(tuple(row)) for row in result)
        finally:
            engine.dispose()
        return "\n".join(rows)


def _migrate(url: str) -> None:
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")


@contextmanager
def runtime_stack(work: Path) -> Iterator[Stack]:
    with ExitStack() as stack:
        postgres = stack.enter_context(PostgresContainer(POSTGRES_IMAGE, driver="psycopg"))
        database_url = postgres.get_connection_url()
        _migrate(database_url)
        api = Settings(database_url=database_url, scheduler_api_key=API_KEY, catalog_path=CATALOG)
        ocr = OcrSettings(mcp_auth_token=MCP_TOKEN, pii_hmac_secret=HMAC_SECRET, host="127.0.0.1")
        rag = RagSettings(mcp_auth_token=MCP_TOKEN, host="127.0.0.1", catalog_path=CATALOG)
        api_url = stack.enter_context(_served(create_app(api)))
        ocr_url = stack.enter_context(_served(build_ocr_app(ocr, TesseractEngine())))
        rag_url = stack.enter_context(_served(build_rag_app(rag, load_catalog(CATALOG))))

        (work / "samples").mkdir(parents=True, exist_ok=True)
        for sample in SAMPLES:
            shutil.copy(REPO_ROOT / "samples" / sample, work / "samples" / sample)
        package = generate(SPEC, work / "generated", allowed_root=work)
        forget_generated_package()
        env = {
            "INPUT_ROOT": str(work),
            "GENERATED_ROOT": str(work / "generated"),
            "OCR_MCP_URL": ocr_url,
            "RAG_MCP_URL": rag_url,
            "MCP_AUTH_TOKEN": MCP_TOKEN,
            "SCHEDULER_API_URL": api_url,
            "SCHEDULER_API_KEY": API_KEY,
            "APP_ENV": "test",
            "AGENT_LLM_BACKEND": "fake",
            "FAKE_LLM_FIXTURE": str(FAKE_LLM_FIXTURES / "pedido_medico_ficticio.json"),
            "LOG_LEVEL": "INFO",
        }
        try:
            yield Stack(work=work, agent_dir=package.path, database_url=database_url, env=env)
        finally:
            forget_generated_package()
            sys.path[:] = [p for p in sys.path if p != str(work / "generated")]


@dataclass
class AgentRun:
    exit_code: int
    report: dict[str, Any]
    logs: str
    """Every log record of the process during the run (agent and servers), JSON lines."""
    sessions: list[Session]
    """Final state of the ADK sessions the run created."""
    llm_requests: list[LlmRequest]
    """Requests the FakeLlm received during the run."""


class _RecordingSessions(InMemorySessionService):
    created: ClassVar[list["_RecordingSessions"]] = []

    def __init__(self) -> None:
        super().__init__()
        _RecordingSessions.created.append(self)


def _fake_llm() -> FakeLlm | None:
    module = sys.modules.get(f"{PACKAGE}.agent")
    model = getattr(getattr(module, "exam_extractor", None), "model", None)
    return model if isinstance(model, FakeLlm) else None


async def run_agent(  # noqa: PLR0913 - the CLI options
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    *,
    fixture: str = "pedido_medico_ficticio.json",
    interactive: bool = False,
    review: cli.ReviewPrompt | None = None,
    log_export: lab_logging.LogExport | None = None,
) -> AgentRun:
    """`lab-agent run` in process against the stack, with logs and sessions captured.

    `log_export` receives the log lines as with the `observability` profile on.
    """
    for name, value in {**stack.env, "FAKE_LLM_FIXTURE": str(FAKE_LLM_FIXTURES / fixture)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "InMemorySessionService", _RecordingSessions)
    _RecordingSessions.created = []
    model = _fake_llm()
    seen = len(model.requests) if model else 0
    logs = io.StringIO()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    lab_logging.configure(
        "agent",
        "INFO",
        stream=logs,
        filters=[LibraryLogFilter(cli.LIBRARY_LOGGERS)],
        export=log_export,
    )
    try:
        outcome = await cli.execute(
            image=image,
            agent_dir=stack.agent_dir,
            spec=SPEC,
            interactive=interactive,
            run_id=str(uuid.uuid4()),
            settings=AgentSettings.from_env(),
            review=review or cli.ask_review,
        )
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
        root.setLevel(level)
        lab_logging._handler = None
        structlog.reset_defaults()
    sessions = []
    for service in _RecordingSessions.created:
        for stored in list(service.sessions.get(PACKAGE, {}).get(cli.USER_ID, {}).values()):
            session = await service.get_session(
                app_name=PACKAGE, user_id=cli.USER_ID, session_id=stored.id
            )
            if session is not None:
                sessions.append(session)
    model = _fake_llm()
    requests = list(model.requests[seen:]) if model else []
    return AgentRun(outcome.exit_code, outcome.report, logs.getvalue(), sessions, requests)
