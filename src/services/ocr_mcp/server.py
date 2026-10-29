import json
import logging
import time
import uuid
from functools import partial

import anyio
import anyio.to_thread
import numpy as np
import structlog
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from lab_common import logging as lab_logging
from lab_common import telemetry as lab_telemetry
from lab_common.contracts.ocr import EngineInfo, OcrLine, OcrResult
from lab_common.mcp_server import BearerAuthMiddleware
from lab_common.text import sanitize
from ocr_mcp.engine import OcrEngine, RecognizedLine
from ocr_mcp.errors import OCR_TIMEOUT_MESSAGE, ErrorCode, OcrError
from ocr_mcp.identity import find_patient_ref
from ocr_mcp.logs import SERVICE, configure_logging
from ocr_mcp.preprocess import preprocess
from ocr_mcp.settings import OcrSettings
from ocr_mcp.validation import validate_image
from pii_guard import PiiRedactor, RedactionResult, document_detectors

TOOL_DESCRIPTION = (
    "Extract the text of a medical order image (PNG or JPEG, base64) with Tesseract. "
    "Personal data is masked with placeholders such as [PESSOA_1] before the text is "
    "returned; patient_ref is a pseudonymous id of the patient."
)
HEALTH_PATH = "/health"
_MAX_LINE_CHARS = 4000
CONFIDENCE_DIGITS = 4

log = lab_logging.get_logger(__name__)


class OcrService:
    def __init__(
        self, settings: OcrSettings, engine: OcrEngine, redactor: PiiRedactor | None = None
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._redactor = redactor or PiiRedactor(document_detectors())
        self._secret = settings.pii_hmac_secret.get_secret_value().encode("utf-8")
        self._limiter = anyio.CapacityLimiter(settings.max_concurrency)

    @property
    def engine(self) -> OcrEngine:
        return self._engine

    async def extract_text(self, image_base64: str, request_id: str) -> OcrResult:
        if not _is_uuid(request_id):
            raise OcrError(
                ErrorCode.INVALID_REQUEST,
                "request_id must be a UUID",
            )

        if self._limiter.available_tokens < 1:
            raise OcrError(
                ErrorCode.BUSY,
                "OCR server is at capacity; retry later",
            )

        deadline = time.monotonic() + self._settings.timeout_s
        try:
            with anyio.fail_after(self._settings.timeout_s):
                # abandon_on_cancel is safe: the engine call enforces its own deadline.
                return await anyio.to_thread.run_sync(
                    partial(self.process, image_base64, deadline),
                    limiter=self._limiter,
                    abandon_on_cancel=True,
                )
        except TimeoutError:
            raise OcrError(
                ErrorCode.OCR_TIMEOUT,
                OCR_TIMEOUT_MESSAGE,
            ) from None

    def process(self, image_base64: str, deadline: float) -> OcrResult:
        """Raw text never leaves this call: it is not returned, logged or attached to errors."""
        settings = self._settings
        image = validate_image(
            image_base64, max_bytes=settings.max_image_bytes, max_pixels=settings.max_pixels
        )

        prepared = preprocess(image.pixels)
        log.info(
            "ocr_preprocess_done",
            duration_ms=prepared.duration_ms,
            skew_degrees=round(prepared.skew_degrees, 2),
            scale=round(prepared.scale, 2),
        )

        started = time.perf_counter()
        raw_lines = self._recognize(prepared.image, deadline)
        ocr_ms = _elapsed_ms(started)

        started = time.perf_counter()
        redaction, patient_ref = self._mask("\n".join(line.text for line in raw_lines))
        pii_ms = _elapsed_ms(started)

        lines = _masked_lines(redaction, raw_lines)
        del raw_lines
        mean_confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        mean_confidence = round(mean_confidence, CONFIDENCE_DIGITS)
        counts = {entity.value: n for entity, n in sorted(redaction.counts.items())}

        log.info("pii_redacted", counts=counts, has_patient_ref=patient_ref is not None)
        log.info(
            "ocr_done",
            lines=len(lines),
            mean_confidence=mean_confidence,
            ocr_ms=ocr_ms,
            pii_ms=pii_ms,
        )

        return OcrResult(
            document_sha256=image.sha256,
            text="\n".join(line.text for line in lines),
            lines=lines,
            mean_confidence=mean_confidence,
            pii_counts=counts,
            patient_ref=patient_ref,
            engine=EngineInfo(name=self._engine.name, lang=self._engine.lang, psm=self._engine.psm),
            timings_ms={"preprocess": prepared.duration_ms, "ocr": ocr_ms, "pii": pii_ms},
        )

    def _recognize(self, image: np.ndarray, deadline: float) -> list[RecognizedLine]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OcrError(
                ErrorCode.OCR_TIMEOUT,
                OCR_TIMEOUT_MESSAGE,
            )

        return _clean_lines(self._engine.recognize(image, timeout_s=remaining))

    def _mask(self, raw_text: str) -> tuple[RedactionResult, str | None]:
        try:
            redaction = self._redactor.redact(raw_text)
            patient_ref = find_patient_ref(self._secret, raw_text, redaction.spans)
        except Exception:  # noqa: BLE001 - any failure while masking must fail closed
            raise OcrError(
                ErrorCode.PII_ENGINE_FAILURE,
                "PII masking failed; no text was released",
            ) from None

        if redaction.has_residual:
            raise OcrError(
                ErrorCode.PII_ENGINE_FAILURE,
                "PII masking left residual data; no text was released",
            )

        return redaction, patient_ref


def build_server(service: OcrService, settings: OcrSettings) -> FastMCP:
    _ensure_json_logging(settings)
    mcp = FastMCP(SERVICE, host=settings.host, port=settings.port, log_level=settings.log_level)

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        engine = service.engine
        try:
            version = await anyio.to_thread.run_sync(engine.version)
        except Exception as exc:  # noqa: BLE001 - report unhealthy instead of failing the probe
            log.error("health_check_failed", error_type=type(exc).__name__)
            return JSONResponse({"status": "unavailable"}, status_code=503)

        return JSONResponse(
            {"status": "ok", "engine": engine.name, "lang": engine.lang, "version": version}
        )

    @mcp.tool(name="extract_text", description=TOOL_DESCRIPTION, structured_output=False)
    async def extract_text(image_base64: str, request_id: str, ctx: Context) -> CallToolResult:
        # Only a well-formed id is logged; anything else is client input.
        logged_id = request_id if _is_uuid(request_id) else None
        with (
            structlog.contextvars.bound_contextvars(request_id=logged_id),
            lab_telemetry.server_span(
                "tools/call extract_text", lab_telemetry.request_meta(ctx), run_id=logged_id
            ),
        ):
            return await _run_extract_text(service, image_base64, request_id)

    return mcp


def build_app(
    settings: OcrSettings, engine: OcrEngine, redactor: PiiRedactor | None = None
) -> ASGIApp:
    service = OcrService(settings, engine, redactor)
    return BearerAuthMiddleware(
        build_server(service, settings).sse_app(),
        token=settings.mcp_auth_token.get_secret_value(),
        public_paths=frozenset({HEALTH_PATH}),
    )


async def _run_extract_text(
    service: OcrService, image_base64: str, request_id: str
) -> CallToolResult:
    started = time.perf_counter()
    log.info("ocr_request", encoded_chars=len(image_base64))

    try:
        result = await service.extract_text(image_base64, request_id)
    except OcrError as error:
        log.warning(
            "ocr_rejected",
            error_code=error.code.value,
            retryable=error.retryable,
            duration_ms=_elapsed_ms(started),
        )
        lab_telemetry.fail(error.code.value)
        return _error_result(error)
    except Exception as exc:  # noqa: BLE001 - fail closed with a generic error
        # Only the type is logged: a message could quote document content.
        log.error(
            "ocr_rejected",
            error_code=ErrorCode.INTERNAL_ERROR.value,
            error_type=type(exc).__name__,
            duration_ms=_elapsed_ms(started),
        )
        lab_telemetry.fail(ErrorCode.INTERNAL_ERROR.value)
        return _error_result(
            OcrError(
                ErrorCode.INTERNAL_ERROR,
                "internal OCR error",
            )
        )

    lab_telemetry.annotate(lines=len(result.lines), pii_masked=sum(result.pii_counts.values()))

    return CallToolResult(
        content=[TextContent(type="text", text=result.model_dump_json())],
        structuredContent=result.model_dump(mode="json"),
    )


def _error_result(error: OcrError) -> CallToolResult:
    body = error.as_dict()
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(body, separators=(",", ":")))],
        structuredContent=body,
        isError=True,
    )


def _ensure_json_logging(settings: OcrSettings) -> None:
    # FastMCP's basicConfig would install a Rich handler without PII redaction.
    if not logging.getLogger().handlers:
        configure_logging(settings.log_level)


def _clean_lines(lines: list[RecognizedLine]) -> list[RecognizedLine]:
    """Zero-width or fullwidth characters inside a CPF would otherwise hide it from detectors."""
    cleaned = []
    for line in lines:
        text = sanitize(line.text.replace("\n", " "), max_chars=_MAX_LINE_CHARS).text
        if text:
            cleaned.append(RecognizedLine(text=text, confidence=line.confidence))

    return cleaned


def _masked_lines(redaction: RedactionResult, raw_lines: list[RecognizedLine]) -> list[OcrLine]:
    # The redactor returns one empty line for an empty text.
    masked_texts = redaction.lines if raw_lines else ()
    return [
        OcrLine(text=masked, confidence=round(line.confidence, CONFIDENCE_DIGITS))
        for masked, line in zip(masked_texts, raw_lines, strict=True)
    ]


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False

    return True


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
