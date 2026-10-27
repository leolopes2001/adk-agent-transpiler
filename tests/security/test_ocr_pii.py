"""No personal data from the sample orders may leave the OCR server.

Runs the real server (Tesseract, SSE, MCP client) and inspects everything it
produces: tool results, its JSON logs and anything written to stdout/stderr.
"""

import base64
import io
import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import structlog
import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent

from lab_common import logging as lab_logging
from lab_common.pii import PiiRedactor, PiiSpan, default_detectors
from ocr_mcp.engine import TesseractEngine
from ocr_mcp.server import OcrService, build_app, build_server
from ocr_mcp.settings import OcrSettings

pytestmark = pytest.mark.security

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = [
    REPO_ROOT / "samples" / "pedido_medico_ficticio.png",
    REPO_ROOT / "samples" / "pedido_injecao.png",
    REPO_ROOT / "samples" / "pedido_ambiguo.png",
]
BOMB = REPO_ROOT / "tests" / "data" / "images" / "decompression_bomb.png"
TOKEN = "security-token-" + "x" * 32
SECRET = "security-secret-" + "y" * 32
CPF = "202.393.613-69"
# Personal data printed on the samples by scripts/make_sample_images.py.
PII_VALUES = [
    "Maria",
    "MARIA",
    "Fictícia da Silva",
    CPF,
    "20239361369",
    "14/03/1985",
    "(31) 90000-0000",
    "90000-0000",
    "0000 1111 2222 3333",
    "João Exemplo",
    "CRM-MG 000000",
]


def settings() -> OcrSettings:
    return OcrSettings(mcp_auth_token=TOKEN, pii_hmac_secret=SECRET, host="127.0.0.1")


@pytest.fixture(scope="module")
def logs() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    lab_logging.configure("ocr-mcp", stream=stream)
    yield stream
    if lab_logging._handler is not None:
        logging.getLogger().removeHandler(lab_logging._handler)
        lab_logging._handler = None
    structlog.reset_defaults()


@pytest.fixture(scope="module")
def server_url(logs: io.StringIO) -> Iterator[str]:
    if shutil.which("tesseract") is None:
        pytest.fail("tesseract is not installed (needs tesseract-ocr and tesseract-ocr-por)")
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(settings(), TesseractEngine()),
            host="127.0.0.1",
            port=0,
            log_config=None,
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@asynccontextmanager
async def session(url: str) -> AsyncIterator[ClientSession]:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with (
        sse_client(f"{url}/sse", headers=headers) as (read, write),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        yield client


async def extract(client: ClientSession, image: bytes) -> CallToolResult:
    return await client.call_tool(
        "extract_text",
        {"image_base64": base64.b64encode(image).decode(), "request_id": str(uuid.uuid4())},
    )


def wire(result: CallToolResult) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def assert_no_pii(text: str) -> None:
    leaked = [value for value in PII_VALUES if value in text]
    assert leaked == []


async def test_results_and_logs_have_no_personal_data(
    server_url: str, logs: io.StringIO, capfd: pytest.CaptureFixture[str]
) -> None:
    async with session(server_url) as client:
        results = [await extract(client, path.read_bytes()) for path in SAMPLES]

    for result in results:
        assert not result.isError
        assert_no_pii(wire(result))
    captured = capfd.readouterr()
    for output in (logs.getvalue(), captured.out, captured.err):
        assert_no_pii(output)


async def test_logs_hold_no_document_text_or_image(server_url: str, logs: io.StringIO) -> None:
    async with session(server_url) as client:
        await extract(client, SAMPLES[0].read_bytes())

    text = logs.getvalue()
    # Not even masked text is logged, nor any part of the image.
    for fragment in ["Hemograma", "[PESSOA_1]", "Laboratório", "iVBORw0KGgo"]:
        assert fragment not in text
    events = {json.loads(line)["event"] for line in text.splitlines()}
    assert {"ocr_request", "pii_redacted", "ocr_done"} <= events


def test_log_redaction_masks_an_injected_cpf(logs: io.StringIO) -> None:
    lab_logging.get_logger("ocr_mcp.server").warning("debug_dump", value=f"CPF {CPF}")
    logging.getLogger("mcp.server").warning("payload %s", CPF)

    tail = logs.getvalue().splitlines()[-2:]
    assert CPF not in "\n".join(tail)
    assert [json.loads(line).get("value", json.loads(line)["event"]) for line in tail] == [
        "CPF [REDACTED:CPF]",
        "payload [REDACTED:CPF]",
    ]


async def test_decompression_bomb_is_rejected_in_under_a_second(server_url: str) -> None:
    async with session(server_url) as client:
        started = time.perf_counter()
        result = await extract(client, BOMB.read_bytes())
        elapsed = time.perf_counter() - started

    assert result.isError
    [content] = result.content
    assert isinstance(content, TextContent)
    assert json.loads(content.text)["code"] == "IMAGE_TOO_MANY_PIXELS"
    assert elapsed < 1.0


class FailingDetector:
    name = "failing"

    def detect(self, text: str) -> list[PiiSpan]:
        raise RuntimeError(text)


async def test_pii_engine_failure_on_real_ocr_releases_nothing(logs: io.StringIO) -> None:
    redactor = PiiRedactor([*default_detectors(), FailingDetector()])
    mcp = build_server(OcrService(settings(), TesseractEngine(), redactor), settings())

    result = await mcp.call_tool(
        "extract_text",
        {
            "image_base64": base64.b64encode(SAMPLES[0].read_bytes()).decode(),
            "request_id": str(uuid.uuid4()),
        },
    )

    assert isinstance(result, CallToolResult)
    assert result.isError
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == "PII_ENGINE_FAILURE"
    assert "Hemograma" not in wire(result)
    assert_no_pii(wire(result))
    assert_no_pii(logs.getvalue())
