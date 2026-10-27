"""Real servers for the runtime integration tests.

Each server runs in a background thread on an ephemeral port, built the way
its `__main__` builds it.
"""

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from starlette.types import ASGIApp

from rag_mcp.catalog import load_catalog
from rag_mcp.server import build_app as build_rag_app
from rag_mcp.settings import RagSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "data" / "exams_catalog.json"
MCP_TOKEN = "runtime-integration-token-" + "x" * 32


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


@pytest.fixture(scope="session")
def mcp_token() -> str:
    return MCP_TOKEN


@pytest.fixture(scope="session")
def rag_url() -> Iterator[str]:
    settings = RagSettings(mcp_auth_token=MCP_TOKEN, host="127.0.0.1", catalog_path=CATALOG_PATH)
    yield from serve(build_rag_app(settings, load_catalog(settings.catalog_path)))
