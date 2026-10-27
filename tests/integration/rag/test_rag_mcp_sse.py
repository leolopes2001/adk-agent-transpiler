"""The RAG server over real HTTP/SSE with a real MCP client.

The server runs in a background thread on an ephemeral port, built exactly as
`python -m rag_mcp` builds it, over the versioned catalog.
"""

import io
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, TextResourceContents
from pydantic import AnyUrl

from lab_common import logging as lab_logging
from lab_common.contracts.rag import BatchSearchResult, CatalogInfo
from rag_mcp.catalog import load_catalog
from rag_mcp.logs import configure_logging
from rag_mcp.server import build_app
from rag_mcp.settings import RagSettings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "data" / "exams_catalog.json"
EXPECTED = json.loads(
    (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
)
TOKEN = "integration-token-" + "x" * 32
# The sample order as the OCR server returns it, then mentions with OCR errors,
# abbreviations, ambiguity and exams the catalog does not have.
QUERIES = [
    "1. Hemograma completo",
    "2. Glicemia de jejum",
    "3. TSH",
    "4. T4 livre",
    "5. Colesterol total e frações",
    "6. Urina tipo 1 (EAS)",
    "7. PCR",
    "Hemogrma completo",
    "Creatinlna",
    "TGO",
    "HbA1c",
    "Vitamina D",
    "Urocultura",
    "Parasitológico de fezes",
    "Ferritina",
    "T4",
    "colesterol",
    "Ressonância magnética de crânio",
    "[PESSOA_1]",
    ";;; ~~",
]


@dataclass
class RunningServer:
    url: str
    logs: io.StringIO


@pytest.fixture(scope="module")
def rag_server() -> Iterator[RunningServer]:
    logs = io.StringIO()
    configure_logging(stream=logs)
    settings = RagSettings(mcp_auth_token=TOKEN, host="127.0.0.1", catalog_path=CATALOG_PATH)
    app = build_app(settings, load_catalog(settings.catalog_path))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = threading.Event()
    for _ in range(100):
        if server.started:
            started.set()
            break
        threading.Event().wait(0.05)
    assert started.is_set(), "server did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield RunningServer(url=f"http://127.0.0.1:{port}", logs=logs)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if lab_logging._handler is not None:
            logging.getLogger().removeHandler(lab_logging._handler)
            lab_logging._handler = None
        structlog.reset_defaults()


@asynccontextmanager
async def mcp_session(server: RunningServer) -> AsyncIterator[ClientSession]:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with (
        sse_client(f"{server.url}/sse", headers=headers) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def payload(result: CallToolResult) -> dict[str, Any]:
    [content] = result.content
    assert isinstance(content, TextContent)
    parsed: dict[str, Any] = json.loads(content.text)
    return parsed


async def test_lists_the_three_tools(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        tools = await session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == ["batch_search", "get_exam", "search_exams"]


async def test_batch_search_keeps_query_order(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        result = await session.call_tool("batch_search", {"queries": QUERIES, "top_k": 5})

    assert not result.isError, payload(result)
    batch = BatchSearchResult.model_validate(payload(result))
    assert result.structuredContent == payload(result)
    assert [r.query for r in batch.results] == QUERIES
    catalog = load_catalog(CATALOG_PATH)
    for item in batch.results:
        assert len(item.candidates) <= 5
        assert all(c.code in catalog and 0 <= c.score <= 1 for c in item.candidates)
    top = [r.candidates[0].code if r.candidates else None for r in batch.results]
    assert top[:7] == [exam["code"] for exam in EXPECTED["exams"]]
    assert top[-1] is None


async def test_sample_order_resolves_from_ocr_lines(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        result = await session.call_tool("batch_search", {"queries": QUERIES[:7], "top_k": 2})

    batch = BatchSearchResult.model_validate(payload(result))
    for item, expected in zip(batch.results, EXPECTED["exams"], strict=True):
        assert item.candidates[0].code == expected["code"], item.query
        assert item.candidates[0].name == expected["name"]


async def test_get_exam(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        found = await session.call_tool("get_exam", {"code": "LAB-10001"})
        missing = await session.call_tool("get_exam", {"code": "LAB-99999"})

    assert payload(found)["name"] == "Hemograma completo"
    assert missing.isError
    assert payload(missing) == {
        "code": "NOT_FOUND",
        "message": "no exam with this code in the catalog",
        "retryable": False,
    }


async def test_catalog_info_resource(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        resource = await session.read_resource(AnyUrl("catalog://info"))

    [content] = resource.contents
    assert isinstance(content, TextResourceContents)
    info = CatalogInfo.model_validate_json(content.text)
    assert info == load_catalog(CATALOG_PATH).info()


async def test_batch_of_fifty_is_fast(rag_server: RunningServer) -> None:
    queries = (QUERIES * 3)[:50]
    async with mcp_session(rag_server) as session:
        await session.call_tool("batch_search", {"queries": queries[:1]})
        started = time.perf_counter()
        result = await session.call_tool("batch_search", {"queries": queries})
        elapsed = time.perf_counter() - started

    assert not result.isError
    assert elapsed < 0.2


async def test_logs_have_no_mentions(rag_server: RunningServer) -> None:
    async with mcp_session(rag_server) as session:
        await session.call_tool("batch_search", {"queries": ["Ressonância magnética de crânio"]})

    logs = rag_server.logs.getvalue()
    assert "Ressonância" not in logs
    assert "ressonancia" not in logs
    assert '"event": "rag_done"' in logs


class TestHttp:
    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-" + "x" * 32}])
    def test_sse_requires_the_token(
        self, rag_server: RunningServer, headers: dict[str, str]
    ) -> None:
        response = httpx.get(f"{rag_server.url}/sse", headers=headers, timeout=5)
        assert response.status_code == 401

    def test_messages_require_the_token(self, rag_server: RunningServer) -> None:
        response = httpx.post(f"{rag_server.url}/messages/?session_id=x", json={}, timeout=5)
        assert response.status_code == 401

    def test_health_is_public(self, rag_server: RunningServer) -> None:
        response = httpx.get(f"{rag_server.url}/health", timeout=5)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["catalog_version"] == "2026.10.0"
        assert body["exam_count"] >= 100
        assert body["catalog_sha256"] == load_catalog(CATALOG_PATH).sha256
