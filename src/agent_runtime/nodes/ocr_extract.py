"""`ocr_extract`: reads the order image and gets its masked text from the OCR server.

The image bytes and their base64 form exist only inside the node function: the
output is the server's `OcrResult` (text already masked, `patient_ref`
already computed by the server) and the state gets hashes and counts.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig
from google.genai import types
from pydantic import Field, ValidationError

from agent_runtime.bootstrap import RuntimeDeps, input_root
from agent_runtime.errors import InvalidInputError, PermanentError, PermanentToolError
from agent_runtime.nodes._common import Params, function_node, request_id, routed
from agent_runtime.schemas import OcrDocument, RunFailure, RunInput
from lab_common import logging as lab_logging
from lab_common import telemetry as lab_telemetry
from lab_common.contracts.ocr import OcrErrorCode, OcrResult

LOW_QUALITY_MESSAGE = (
    "Não foi possível ler o pedido com qualidade suficiente. "
    "Envie uma imagem mais nítida, sem cortes e bem iluminada."
)
_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
_INPUT_TOOL_ERRORS = frozenset(
    {
        OcrErrorCode.INVALID_BASE64,
        OcrErrorCode.IMAGE_TOO_LARGE,
        OcrErrorCode.UNSUPPORTED_FORMAT,
        OcrErrorCode.IMAGE_TOO_MANY_PIXELS,
        OcrErrorCode.CORRUPTED_IMAGE,
    }
)

log = lab_logging.get_logger(__name__)


class OcrExtractParams(Params):
    min_mean_confidence: Annotated[float, Field(ge=0, le=1)] = 0.55
    max_image_bytes: Annotated[int, Field(ge=1)] = 8 * 1024 * 1024


def load_image(image_path: str, root: Path, max_bytes: int) -> bytes:
    """Bytes of a PNG or JPEG file under `root`, or InvalidInputError.

    Error messages never repeat the path: a file name can hold a patient's name.
    """
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise InvalidInputError("IMAGE_NOT_FOUND", "image file not found") from None
    if not resolved.is_relative_to(root):
        raise InvalidInputError("IMAGE_OUTSIDE_INPUT_ROOT", "image is outside INPUT_ROOT")
    if not resolved.is_file():
        raise InvalidInputError("IMAGE_NOT_A_FILE", "image path is not a regular file")
    size = resolved.stat().st_size
    if size == 0:
        raise InvalidInputError("IMAGE_EMPTY", "image file is empty")
    if size > max_bytes:
        raise InvalidInputError("IMAGE_TOO_LARGE", f"image is larger than {max_bytes} bytes")
    with resolved.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise InvalidInputError("IMAGE_TOO_LARGE", f"image is larger than {max_bytes} bytes")
    if not data.startswith(_MAGIC):
        raise InvalidInputError("UNSUPPORTED_FORMAT", "image must be PNG or JPEG")
    return data


def parse_run_input(node_input: object) -> RunInput:
    """The run input, given as the user message (JSON text), a dict or a RunInput."""
    if isinstance(node_input, RunInput):
        return node_input
    value: Any = node_input
    if isinstance(node_input, types.Content):
        value = "".join(part.text or "" for part in node_input.parts or [])
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return RunInput.model_validate(value)
    except (ValueError, ValidationError):
        raise InvalidInputError("INVALID_RUN_INPUT", "expected {'image_path': str}") from None


def _validate_result(body: object, sha256: str) -> OcrResult:
    try:
        result = OcrResult.model_validate(body)
    except ValidationError:
        raise PermanentError("OCR_INVALID_RESULT", "OCR result breaks the contract") from None
    if result.document_sha256 != sha256:
        raise PermanentError("OCR_DOCUMENT_MISMATCH", "OCR result is for another document")
    return result


def ocr_extract(  # noqa: PLR0913 - one keyword per node param
    *,
    name: str,
    deps: RuntimeDeps,
    server: str,
    tool: str,
    min_mean_confidence: float = 0.55,
    max_image_bytes: int = 8 * 1024 * 1024,
    retry: RetryConfig | None = None,
    timeout: float | None = None,
) -> FunctionNode:
    """Reads the image and calls the OCR tool; routes `ok` or `low_quality`."""
    params = OcrExtractParams(
        min_mean_confidence=min_mean_confidence, max_image_bytes=max_image_bytes
    )

    async def run(ctx: Context, node_input: Any) -> Event:
        run_input = parse_run_input(node_input)
        image = load_image(run_input.image_path, input_root(), params.max_image_bytes)
        sha256 = hashlib.sha256(image).hexdigest()
        args = {
            "image_base64": base64.b64encode(image).decode("ascii"),
            "request_id": request_id(ctx),
        }
        del image
        try:
            body = await deps.mcp.call(server, tool, args)
        except PermanentToolError as error:
            if error.tool_code in _INPUT_TOOL_ERRORS:
                raise InvalidInputError("INVALID_IMAGE", error.tool_code) from None
            raise
        finally:
            del args
        result: OcrDocument = _validate_result(body, sha256)

        ctx.state["run.document_sha256"] = result.document_sha256
        ctx.state["run.patient_ref"] = result.patient_ref
        ctx.state["run.pii_counts"] = dict(result.pii_counts)
        low_quality = result.mean_confidence < params.min_mean_confidence or not result.text.strip()
        log.info(
            "ocr_result",
            node=name,
            lines=len(result.lines),
            mean_confidence=result.mean_confidence,
            pii_counts=dict(result.pii_counts),
            has_patient_ref=result.patient_ref is not None,
            low_quality=low_quality,
        )
        lab_telemetry.annotate(lines=len(result.lines), pii_masked=sum(result.pii_counts.values()))
        if low_quality:
            failure = RunFailure(code="OCR_LOW_QUALITY", user_message=LOW_QUALITY_MESSAGE)
            return routed(failure, "low_quality")
        return routed(result, "ok")

    return function_node(name, run, retry=retry, timeout=timeout)
