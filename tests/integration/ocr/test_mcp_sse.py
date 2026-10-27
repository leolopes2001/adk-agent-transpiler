"""The OCR server over real HTTP/SSE with a real MCP client and real Tesseract.

The server runs in a background thread on an ephemeral port, built exactly as
`python -m ocr_mcp` builds it.
"""

import base64
import io
import json
import logging
import re
import shutil
import threading
import uuid
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
from mcp.types import CallToolResult, TextContent
from rapidfuzz import fuzz

from lab_common import logging as lab_logging
from lab_common.contracts.ocr import OcrResult
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.server import build_app
from ocr_mcp.settings import OcrSettings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = REPO_ROOT / "samples" / "pedido_medico_ficticio.png"
EXPECTED = json.loads(
    (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
)
TOKEN = "integration-token-" + "x" * 32
SECRET = "integration-secret-" + "y" * 32
MIN_FUZZY_SCORE = 85
# Personal data printed on the samples by scripts/make_sample_images.py.
PII_VALUES = [
    "Maria",
    "Fictícia da Silva",
    "202.393.613-69",
    "20239361369",
    "14/03/1985",
    "90000-0000",
    "0000 1111 2222 3333",
    "João Exemplo",
    "000000",
]


@dataclass
class RunningServer:
    url: str
    logs: io.StringIO


@pytest.fixture(scope="module")
def ocr_server() -> Iterator[RunningServer]:
    if shutil.which("tesseract") is None:
        pytest.fail("tesseract is not installed (needs tesseract-ocr and tesseract-ocr-por)")
    logs = io.StringIO()
    lab_logging.configure("ocr-mcp", stream=logs)
    settings = OcrSettings(mcp_auth_token=TOKEN, pii_hmac_secret=SECRET, host="127.0.0.1")
    app = build_app(settings, TesseractEngine())
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


async def extract(server: RunningServer, image: bytes | str) -> CallToolResult:
    encoded = image if isinstance(image, str) else base64.b64encode(image).decode("ascii")
    async with mcp_session(server) as session:
        return await session.call_tool(
            "extract_text", {"image_base64": encoded, "request_id": str(uuid.uuid4())}
        )


def payload(result: CallToolResult) -> dict[str, Any]:
    [content] = result.content
    assert isinstance(content, TextContent)
    parsed: dict[str, Any] = json.loads(content.text)
    return parsed


async def test_lists_only_extract_text(ocr_server: RunningServer) -> None:
    async with mcp_session(ocr_server) as session:
        tools = await session.list_tools()

    assert [tool.name for tool in tools.tools] == ["extract_text"]


async def test_sample_order_returns_masked_text_with_every_exam(
    ocr_server: RunningServer,
) -> None:
    result = await extract(ocr_server, SAMPLE.read_bytes())

    assert not result.isError, payload(result)
    ocr = OcrResult.model_validate(payload(result))
    assert result.structuredContent == payload(result)
    for exam in EXPECTED["exams"]:
        best = max(fuzz.partial_ratio(exam["mention"], line.text) for line in ocr.lines)
        assert best >= MIN_FUZZY_SCORE, exam["mention"]
    assert ocr.mean_confidence >= 0.7
    assert ocr.patient_ref is not None
    assert re.fullmatch(r"^pt_[A-Z2-7]{26}$", ocr.patient_ref)
    assert ocr.pii_counts["PERSON"] == 2
    assert ocr.pii_counts["CPF"] == 1
    assert "Paciente: [PESSOA_1]" in ocr.text


async def test_sample_order_output_has_no_personal_data(ocr_server: RunningServer) -> None:
    result = await extract(ocr_server, SAMPLE.read_bytes())

    wire = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    for value in PII_VALUES:
        assert value not in wire


async def test_same_document_same_patient_ref(ocr_server: RunningServer) -> None:
    first = payload(await extract(ocr_server, SAMPLE.read_bytes()))
    second = payload(await extract(ocr_server, SAMPLE.read_bytes()))

    assert first["patient_ref"] == second["patient_ref"]
    assert first["document_sha256"] == second["document_sha256"]


async def test_errors_are_json_tool_errors(ocr_server: RunningServer) -> None:
    result = await extract(ocr_server, "not base64!")

    assert result.isError
    assert payload(result) == {
        "code": "INVALID_BASE64",
        "message": "image_base64 is not valid base64",
        "retryable": False,
    }


class TestHttp:
    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-" + "x" * 32}])
    def test_sse_requires_the_token(
        self, ocr_server: RunningServer, headers: dict[str, str]
    ) -> None:
        response = httpx.get(f"{ocr_server.url}/sse", headers=headers, timeout=5)
        assert response.status_code == 401

    def test_messages_require_the_token(self, ocr_server: RunningServer) -> None:
        response = httpx.post(f"{ocr_server.url}/messages/?session_id=x", json={}, timeout=5)
        assert response.status_code == 401

    def test_health_is_public(self, ocr_server: RunningServer) -> None:
        response = httpx.get(f"{ocr_server.url}/health", timeout=10)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert (body["engine"], body["lang"]) == ("tesseract", "por")
        assert re.match(r"^\d+\.\d+", body["version"])
