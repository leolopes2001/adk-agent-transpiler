"""MCP server (SSE transport) with the `extract_text` tool.

This is the only process that ever holds raw OCR text. `OcrService.process`
reads the image, recognizes and masks it in one synchronous call, so raw text
exists only in that call's locals: it is never returned, logged, written to
disk or attached to an exception. If masking fails or leaves anything behind
(the redactor's re-scan), the call fails with `PII_ENGINE_FAILURE` and no text.

Every path except `/health` requires `Authorization: Bearer <MCP_AUTH_TOKEN>`;
the check wraps the whole SSE app, so both `/sse` and `/messages/` are covered.

Errors are tool results with `isError: true` and a JSON body
`{code, message, retryable}` (`OcrToolError`). They are built here instead of
raised, because FastMCP would prefix the message of a raised exception.
"""

import hmac
import json
import logging
import time
import uuid
from functools import partial

import anyio
import anyio.to_thread
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from lab_common import logging as lab_logging
from lab_common.contracts.ocr import EngineInfo, OcrLine, OcrResult
from lab_common.pii import PiiRedactor, default_detectors
from lab_common.text import sanitize
from ocr_mcp.engine import OcrEngine, RecognizedLine
from ocr_mcp.errors import ErrorCode, OcrError
from ocr_mcp.identity import find_patient_ref
from ocr_mcp.preprocess import preprocess
from ocr_mcp.settings import OcrSettings
from ocr_mcp.validation import validate_image

SERVICE = "ocr-mcp"
TOOL_DESCRIPTION = (
    "Extract the text of a medical order image (PNG or JPEG, base64) with Tesseract. "
    "Personal data is masked with placeholders such as [PESSOA_1] before the text is "
    "returned; patient_ref is a pseudonymous id of the patient."
)
_MAX_LINE_CHARS = 4000

log = lab_logging.get_logger(__name__)


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _clean_lines(lines: list[RecognizedLine]) -> list[RecognizedLine]:
    """Normalize OCR lines the way untrusted text is normalized everywhere else.

    Zero-width or fullwidth characters inside a CPF would otherwise hide it
    from the detectors; lines that end up empty are dropped.
    """
    cleaned = []
    for line in lines:
        text = sanitize(line.text.replace("\n", " "), max_chars=_MAX_LINE_CHARS).text
        if text:
            cleaned.append(RecognizedLine(text=text, confidence=line.confidence))
    return cleaned


class OcrService:
    def __init__(
        self, settings: OcrSettings, engine: OcrEngine, redactor: PiiRedactor | None = None
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._redactor = redactor or PiiRedactor(default_detectors())
        self._secret = settings.pii_hmac_secret.get_secret_value().encode("utf-8")
        self._limiter = anyio.CapacityLimiter(settings.max_concurrency)

    @property
    def engine(self) -> OcrEngine:
        return self._engine

    async def extract_text(self, image_base64: str, request_id: str) -> OcrResult:
        if not _is_uuid(request_id):
            raise OcrError(ErrorCode.INVALID_REQUEST, "request_id must be a UUID")
        if self._limiter.available_tokens < 1:
            raise OcrError(ErrorCode.BUSY, "OCR server is at capacity; retry later")
        deadline = time.monotonic() + self._settings.timeout_s
        try:
            with anyio.fail_after(self._settings.timeout_s):
                # abandon_on_cancel: the timeout returns to the client at once; the
                # worker stops by itself, since the engine call has its own deadline.
                return await anyio.to_thread.run_sync(
                    partial(self.process, image_base64, deadline),
                    limiter=self._limiter,
                    abandon_on_cancel=True,
                )
        except TimeoutError:
            raise OcrError(ErrorCode.OCR_TIMEOUT, "OCR exceeded the time limit") from None

    def process(self, image_base64: str, deadline: float) -> OcrResult:
        """Validate, recognize and mask one image; raw text never leaves this call."""
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OcrError(ErrorCode.OCR_TIMEOUT, "OCR exceeded the time limit")
        raw_lines = _clean_lines(self._engine.recognize(prepared.image, timeout_s=remaining))
        ocr_ms = _ms(started)

        started = time.perf_counter()
        raw_text = "\n".join(line.text for line in raw_lines)
        try:
            redaction = self._redactor.redact(raw_text)
            patient_ref = find_patient_ref(self._secret, raw_text, redaction.spans)
        except Exception:  # noqa: BLE001 - any failure while masking must fail closed
            raise OcrError(
                ErrorCode.PII_ENGINE_FAILURE, "PII masking failed; no text was released"
            ) from None
        if redaction.has_residual:
            raise OcrError(
                ErrorCode.PII_ENGINE_FAILURE, "PII masking left residual data; no text was released"
            )
        pii_ms = _ms(started)
        del raw_text

        # An empty document still yields one (empty) masked line.
        masked_lines = redaction.lines if raw_lines else ()
        lines = [
            OcrLine(text=masked, confidence=round(line.confidence, 4))
            for masked, line in zip(masked_lines, raw_lines, strict=True)
        ]
        del raw_lines
        mean_confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        counts = {entity.value: n for entity, n in sorted(redaction.counts.items())}
        log.info("pii_redacted", counts=counts, has_patient_ref=patient_ref is not None)
        log.info(
            "ocr_done",
            lines=len(lines),
            mean_confidence=round(mean_confidence, 4),
            ocr_ms=ocr_ms,
            pii_ms=pii_ms,
        )
        return OcrResult(
            document_sha256=image.sha256,
            text="\n".join(line.text for line in lines),
            lines=lines,
            mean_confidence=round(mean_confidence, 4),
            pii_counts=counts,
            patient_ref=patient_ref,
            engine=EngineInfo(name=self._engine.name, lang=self._engine.lang, psm=self._engine.psm),
            timings_ms={"preprocess": prepared.duration_ms, "ocr": ocr_ms, "pii": pii_ms},
        )


def _error_result(error: OcrError) -> CallToolResult:
    body = error.as_dict()
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(body, separators=(",", ":")))],
        structuredContent=body,
        isError=True,
    )


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


async def _call(service: OcrService, image_base64: str, request_id: str) -> CallToolResult:
    started = time.perf_counter()
    log.info("ocr_request", encoded_chars=len(image_base64))
    try:
        result = await service.extract_text(image_base64, request_id)
    except OcrError as error:
        log.warning(
            "ocr_rejected",
            error_code=error.code.value,
            retryable=error.retryable,
            duration_ms=_ms(started),
        )
        return _error_result(error)
    except Exception as exc:  # noqa: BLE001 - fail closed with a generic error
        # Only the type is logged: a message could quote document content.
        log.error(
            "ocr_rejected",
            error_code=ErrorCode.INTERNAL_ERROR.value,
            error_type=type(exc).__name__,
            duration_ms=_ms(started),
        )
        return _error_result(OcrError(ErrorCode.INTERNAL_ERROR, "internal OCR error"))
    return CallToolResult(
        content=[TextContent(type="text", text=result.model_dump_json())],
        structuredContent=result.model_dump(mode="json"),
    )


def _ensure_json_logging(settings: OcrSettings) -> None:
    # FastMCP calls logging.basicConfig with a Rich handler, which would print
    # records without the redaction processor; it is a no-op once the root
    # logger has a handler.
    if not logging.getLogger().handlers:
        lab_logging.configure(SERVICE, settings.log_level)


class BearerAuthMiddleware:
    """Rejects HTTP requests without the bearer token, except on public paths.

    The token is compared in constant time; the 401 body and the log line never
    include what the client sent.
    """

    def __init__(self, app: ASGIApp, token: str, public_paths: frozenset[str]) -> None:
        self._app = app
        self._expected = f"bearer {token}".encode()
        self._public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._public_paths:
            await self._app(scope, receive, send)
            return
        if not self._authorized(scope):
            log.warning("auth_failed", path=scope["path"] if scope["path"] in _MCP_PATHS else None)
            response = JSONResponse(
                {"code": "UNAUTHORIZED", "message": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        values = [v for k, v in scope["headers"] if k == b"authorization"]
        if len(values) != 1:
            return False
        scheme, _, credentials = values[0].partition(b" ")
        # The scheme is case-insensitive (RFC 9110); the token is not.
        return hmac.compare_digest(scheme.lower() + b" " + credentials, self._expected)


_MCP_PATHS = frozenset({"/sse", "/messages/"})
HEALTH_PATH = "/health"


def build_server(service: OcrService, settings: OcrSettings) -> FastMCP:
    _ensure_json_logging(settings)
    mcp = FastMCP(SERVICE, host=settings.host, port=settings.port, log_level=settings.log_level)

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> Response:
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
    async def extract_text(image_base64: str, request_id: str) -> CallToolResult:
        # Only a well-formed id is logged; anything else is client input.
        logged_id = request_id if _is_uuid(request_id) else None
        with structlog.contextvars.bound_contextvars(request_id=logged_id):
            return await _call(service, image_base64, request_id)

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
