import base64
import io
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from ocr_fakes import SECRET, FakeOcrEngine, make_settings, png_base64

from lab_common import logging as lab_logging
from lab_common.contracts.ocr import OcrResult
from lab_common.pii import PiiRedactor, PiiSpan, default_detectors
from ocr_mcp.engine import RecognizedLine
from ocr_mcp.errors import ErrorCode, OcrError
from ocr_mcp.server import OcrService, build_server
from ocr_mcp.settings import OcrSettings

pytestmark = pytest.mark.unit

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"
# Lines as Tesseract reads samples/pedido_medico_ficticio.png; personal data
# is the one printed by scripts/make_sample_images.py.
SAMPLE_LINES = [
    ("Laboratório Exemplo (fictício)", 0.96),
    ("Rua Imaginária, 100 — Belo Horizonte/MG", 0.94),
    ("SOLICITAÇÃO DE EXAMES LABORATORIAIS", 0.97),
    ("Paciente: Maria Fictícia da Silva", 0.95),
    ("CPF: 202.393.613-69", 0.92),
    ("Nascimento: 14/03/1985", 0.96),
    ("Telefone: (31) 90000-0000", 0.93),
    ("Convênio: Plano Exemplo Saúde (fictício)", 0.97),
    ("Carteirinha: 0000 1111 2222 3333", 0.95),
    ("Exames solicitados:", 0.96),
    ("1. Hemograma completo", 0.96),
    ("2. Glicemia de jejum", 0.96),
    ("3. TSH", 0.95),
    ("4. T4 livre", 0.89),
    ("5. Colesterol total e frações", 0.96),
    ("6. Urina tipo 1 (EAS)", 0.94),
    ("7. PCR", 0.96),
    ("Indicação clínica: avaliação de rotina.", 0.96),
    ("Dr. João Exemplo — CRM-MG 000000", 0.89),
    ("Belo Horizonte, 20/10/2026", 0.97),
]
PII_VALUES = [
    "Maria",
    "Fictícia da Silva",
    "202.393.613-69",
    "20239361369",
    "14/03/1985",
    "90000-0000",
    "0000 1111 2222 3333",
    "João",
    "000000",
    "Rua Imaginária",
]


def sample_engine() -> FakeOcrEngine:
    return FakeOcrEngine([RecognizedLine(text, conf) for text, conf in SAMPLE_LINES])


@pytest.fixture(autouse=True)
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    lab_logging.configure("ocr-mcp", stream=stream)
    yield stream
    root = logging.getLogger()
    if lab_logging._handler is not None:
        root.removeHandler(lab_logging._handler)
        lab_logging._handler = None
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def server(
    engine: FakeOcrEngine | Any | None = None,
    settings: OcrSettings | None = None,
    redactor: PiiRedactor | None = None,
) -> FastMCP:
    settings = settings or make_settings()
    service = OcrService(settings, engine or sample_engine(), redactor)
    return build_server(service, settings)


async def call(
    mcp: FastMCP, image_base64: str | None = None, request_id: str | None = None
) -> CallToolResult:
    result = await mcp.call_tool(
        "extract_text",
        {
            "image_base64": png_base64() if image_base64 is None else image_base64,
            "request_id": request_id or str(uuid.uuid4()),
        },
    )
    assert isinstance(result, CallToolResult)
    return result


def body(result: CallToolResult) -> dict[str, Any]:
    [content] = result.content
    assert isinstance(content, TextContent)
    parsed: dict[str, Any] = json.loads(content.text)
    assert parsed == result.structuredContent
    return parsed


def ok(result: CallToolResult) -> OcrResult:
    assert not result.isError, body(result)
    return OcrResult.model_validate(body(result))


def error(result: CallToolResult) -> dict[str, Any]:
    assert result.isError
    return body(result)


def raw_output(result: CallToolResult) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


class TestSampleOrder:
    async def test_returns_masked_text_without_any_personal_value(self) -> None:
        result = await call(server())

        output = raw_output(result)
        for value in PII_VALUES:
            assert value not in output

    async def test_keeps_exam_lines_and_structure(self) -> None:
        result = ok(await call(server()))

        assert "Paciente: [PESSOA_1]" in result.text
        assert "Dr. [PESSOA_2] — [CRM_1]" in result.text
        for exam in ["Hemograma completo", "Glicemia de jejum", "TSH", "T4 livre", "7. PCR"]:
            assert exam in result.text

    async def test_lines_match_text_and_keep_confidences(self) -> None:
        result = ok(await call(server()))

        assert [line.text for line in result.lines] == result.text.split("\n")
        assert [line.confidence for line in result.lines] == [c for _, c in SAMPLE_LINES]
        assert result.mean_confidence == pytest.approx(
            sum(c for _, c in SAMPLE_LINES) / len(SAMPLE_LINES), abs=1e-4
        )

    async def test_reports_counts_engine_and_timings(self) -> None:
        result = ok(await call(server()))

        assert result.contract_version == "1"
        assert result.pii_counts == {
            "ADDRESS": 1,
            "BIRTH_DATE": 1,
            "CPF": 1,
            "CRM": 1,
            "HEALTH_CARD": 1,
            "PERSON": 2,
            "PHONE": 1,
        }
        assert result.engine.model_dump() == {"name": "fake", "lang": "por", "psm": 6}
        assert set(result.timings_ms) == {"preprocess", "ocr", "pii"}

    async def test_patient_ref_is_stable_and_matches_the_api_format(self) -> None:
        mcp = server()
        first = ok(await call(mcp))
        second = ok(await call(mcp))

        assert first.patient_ref is not None
        assert re.fullmatch(r"^pt_[A-Z2-7]{26}$", first.patient_ref)
        assert first.patient_ref == second.patient_ref

    async def test_patient_ref_depends_on_the_secret(self) -> None:
        first = ok(await call(server()))
        other = ok(await call(server(settings=make_settings(pii_hmac_secret="z" * 40))))
        assert first.patient_ref != other.patient_ref

    async def test_document_hash_is_of_the_decoded_image(self) -> None:
        import hashlib  # noqa: PLC0415

        image = png_base64()
        result = ok(await call(server(), image_base64=image))
        assert result.document_sha256 == hashlib.sha256(base64.b64decode(image)).hexdigest()

    async def test_logs_counts_but_no_text(self, log_stream: io.StringIO) -> None:
        request_id = str(uuid.uuid4())
        await call(server(), request_id=request_id)

        logs = log_stream.getvalue()
        for value in [*PII_VALUES, "Hemograma", "[PESSOA_1]"]:
            assert value not in logs
        records = [json.loads(line) for line in logs.splitlines()]
        events = [r["event"] for r in records]
        assert events == ["ocr_request", "ocr_preprocess_done", "pii_redacted", "ocr_done"]
        assert all(r["request_id"] == request_id for r in records)
        assert records[2]["counts"]["PERSON"] == 2


class TestEdgeCases:
    async def test_blank_page(self) -> None:
        result = ok(await call(server(FakeOcrEngine([]))))

        assert (result.text, result.lines, result.mean_confidence) == ("", [], 0.0)
        assert result.patient_ref is None
        assert result.pii_counts == {}

    async def test_order_without_patient_identifier(self) -> None:
        engine = FakeOcrEngine([RecognizedLine("1. Hemograma completo", 0.9)])
        assert ok(await call(server(engine))).patient_ref is None

    async def test_invisible_characters_cannot_hide_a_cpf(self) -> None:
        engine = FakeOcrEngine([RecognizedLine("CPF: 202.393\u200b.613-69", 0.9)])
        result = await call(server(engine))

        assert ok(result).text == "CPF: [CPF_1]"

    async def test_lines_emptied_by_normalization_are_dropped(self) -> None:
        engine = FakeOcrEngine([RecognizedLine("\u200b", 0.1), RecognizedLine("TSH", 0.9)])
        result = ok(await call(server(engine)))
        assert [line.text for line in result.lines] == ["TSH"]

    async def test_engine_receives_the_preprocessed_image(self) -> None:
        engine = sample_engine()
        await call(server(engine), image_base64=png_base64((40, 30)))
        # Narrow images are upscaled (capped at 4x) before OCR.
        assert engine.calls == [(120, 160)]


class FailingDetector:
    name = "failing"

    def detect(self, text: str) -> list[PiiSpan]:
        raise RuntimeError(f"cannot handle {text}")


class TestFailClosed:
    async def test_pii_engine_failure_returns_no_text(self) -> None:
        redactor = PiiRedactor([*default_detectors(), FailingDetector()])

        result = await call(server(redactor=redactor))

        assert error(result) == {
            "code": "PII_ENGINE_FAILURE",
            "message": "PII masking failed; no text was released",
            "retryable": False,
        }
        output = raw_output(result)
        assert "Hemograma" not in output
        for value in PII_VALUES:
            assert value not in output

    async def test_residual_after_masking_returns_no_text(self) -> None:
        class Sticky:
            name = "sticky"

            def detect(self, text: str) -> list[PiiSpan]:
                from lab_common.pii import PiiEntity  # noqa: PLC0415

                return [PiiSpan(PiiEntity.PERSON, 0, 1, 1.0, "sticky")] if text else []

        result = await call(server(redactor=PiiRedactor([Sticky()])))

        assert error(result)["code"] == "PII_ENGINE_FAILURE"
        assert "Laborat" not in raw_output(result)

    async def test_unexpected_error_is_generic_and_logs_only_the_type(
        self, log_stream: io.StringIO
    ) -> None:
        class Broken(FakeOcrEngine):
            def recognize(self, image: Any, timeout_s: float) -> list[RecognizedLine]:
                raise ValueError("Paciente: Maria Fictícia da Silva 202.393.613-69")

        result = await call(server(Broken()))

        assert error(result) == {
            "code": "INTERNAL_ERROR",
            "message": "internal OCR error",
            "retryable": False,
        }
        logs = log_stream.getvalue()
        assert "Maria" not in logs
        assert '"error_type": "ValueError"' in logs
        assert "Traceback" not in logs

    async def test_pii_failure_logs_no_text(self, log_stream: io.StringIO) -> None:
        redactor = PiiRedactor([FailingDetector()])
        await call(server(redactor=redactor))

        logs = log_stream.getvalue()
        assert "PII_ENGINE_FAILURE" in logs
        for value in [*PII_VALUES, "Hemograma"]:
            assert value not in logs


class TestInputErrors:
    @pytest.mark.parametrize(
        ("image", "code"),
        [
            ("%%%", "INVALID_BASE64"),
            (base64.b64encode(b"GIF89a....").decode(), "UNSUPPORTED_FORMAT"),
            (
                base64.b64encode((IMAGES / "decompression_bomb.png").read_bytes()).decode(),
                "IMAGE_TOO_MANY_PIXELS",
            ),
            (
                base64.b64encode((IMAGES / "corrupted.png").read_bytes()).decode(),
                "CORRUPTED_IMAGE",
            ),
        ],
    )
    async def test_validation_errors(self, image: str, code: str) -> None:
        result = await call(server(), image_base64=image)
        assert (error(result)["code"], error(result)["retryable"]) == (code, False)

    async def test_image_too_large(self) -> None:
        result = await call(server(settings=make_settings(max_image_bytes=20)), png_base64())
        assert error(result)["code"] == "IMAGE_TOO_LARGE"

    async def test_request_id_must_be_a_uuid(self, log_stream: io.StringIO) -> None:
        result = await call(server(), request_id="Maria Fictícia")

        assert error(result)["code"] == "INVALID_REQUEST"
        assert "Maria" not in log_stream.getvalue()
        assert json.loads(log_stream.getvalue().splitlines()[0])["request_id"] is None


class BlockingEngine(FakeOcrEngine):
    def __init__(self) -> None:
        super().__init__([RecognizedLine("TSH", 0.9)])
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize(self, image: Any, timeout_s: float) -> list[RecognizedLine]:
        self.started.set()
        self.release.wait(timeout=5)
        return super().recognize(image, timeout_s)


class TestCapacityAndTimeout:
    async def test_busy_when_all_slots_are_taken(self) -> None:
        engine = BlockingEngine()
        mcp = server(engine, settings=make_settings(max_concurrency=1))
        results: list[CallToolResult] = []

        async def first() -> None:
            results.append(await call(mcp))

        async with anyio.create_task_group() as tg:
            tg.start_soon(first)
            await anyio.to_thread.run_sync(engine.started.wait, 5)
            busy = await call(mcp)
            engine.release.set()

        assert error(busy) == {
            "code": "BUSY",
            "message": "OCR server is at capacity; retry later",
            "retryable": True,
        }
        assert ok(results[0]).text == "TSH"

    async def test_timeout_returns_retryable_error_without_waiting(self) -> None:
        engine = BlockingEngine()
        mcp = server(engine, settings=make_settings(timeout_s=0.3))

        started = time.perf_counter()
        result = await call(mcp)
        elapsed = time.perf_counter() - started
        engine.release.set()

        assert error(result) == {
            "code": "OCR_TIMEOUT",
            "message": "OCR exceeded the time limit",
            "retryable": True,
        }
        assert elapsed < 2

    async def test_engine_timeout_is_reported(self) -> None:
        class Slow(FakeOcrEngine):
            def recognize(self, image: Any, timeout_s: float) -> list[RecognizedLine]:
                raise OcrError(ErrorCode.OCR_TIMEOUT, "OCR exceeded the time limit")

        assert error(await call(server(Slow())))["code"] == "OCR_TIMEOUT"

    async def test_engine_gets_the_remaining_time(self) -> None:
        seen: list[float] = []

        class Recording(FakeOcrEngine):
            def recognize(self, image: Any, timeout_s: float) -> list[RecognizedLine]:
                seen.append(timeout_s)
                return []

        await call(server(Recording(), settings=make_settings(timeout_s=5)))
        assert 0 < seen[0] <= 5


async def test_lists_only_extract_text() -> None:
    tools = await server().list_tools()

    assert [tool.name for tool in tools] == ["extract_text"]
    assert sorted(tools[0].inputSchema["required"]) == ["image_base64", "request_id"]
    assert tools[0].outputSchema is None


def test_secret_is_used_as_bytes() -> None:
    assert len(SECRET.encode()) >= 32
