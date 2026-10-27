"""Real servers for the runtime integration tests (see tests/runtime_stack.py)."""

from collections.abc import Iterator

import pytest

from rag_mcp.catalog import load_catalog
from rag_mcp.server import build_app as build_rag_app
from rag_mcp.settings import RagSettings
from runtime_stack import CATALOG, MCP_TOKEN, serve


@pytest.fixture(scope="session")
def mcp_token() -> str:
    return MCP_TOKEN


@pytest.fixture(scope="session")
def rag_url() -> Iterator[str]:
    settings = RagSettings(mcp_auth_token=MCP_TOKEN, host="127.0.0.1", catalog_path=CATALOG)
    yield from serve(build_rag_app(settings, load_catalog(settings.catalog_path)))
