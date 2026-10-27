import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from google.adk.workflow import RetryConfig
from google.genai import types
from runtime_fakes import FakeMcp, run_node

from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.errors import (
    InvalidInputError,
    PermanentError,
    PermanentToolError,
    TransientError,
    TransientToolError,
)
from agent_runtime.nodes.ocr_extract import load_image, ocr_extract, parse_run_input
from agent_runtime.schemas import RunFailure, RunInput
from lab_common.contracts.ocr import OcrResult

pytestmark = pytest.mark.unit

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
RUN_ID = "2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11"
PATIENT_REF = "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"
MASKED = "Paciente: [PESSOA_1]\nCPF: [CPF_1]\n1. Hemograma completo\n3. TSH"


def ocr_body(image_base64: str, *, confidence: float = 0.95, text: str = MASKED) -> dict[str, Any]:
    lines = [{"text": t, "confidence": confidence} for t in text.split("\n")] if text else []
    return {
        "contract_version": "1",
        "document_sha256": hashlib.sha256(base64.b64decode(image_base64)).hexdigest(),
        "text": text,
        "lines": lines,
        "mean_confidence": confidence if text else 0.0,
        "pii_counts": {"CPF": 1, "PERSON": 1},
        "patient_ref": PATIENT_REF,
        "engine": {"name": "tesseract", "lang": "por", "psm": 6},
        "timings_ms": {"ocr": 10},
    }


def answer(**kwargs: Any) -> Any:
    def respond(args: Mapping[str, Any]) -> dict[str, Any]:
        return ocr_body(args["image_base64"], **kwargs)

    return respond


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INPUT_ROOT", str(tmp_path))
    (tmp_path / "order.png").write_bytes(PNG)
    return tmp_path.resolve()


@pytest.fixture
def fake_mcp() -> FakeMcp:
    return FakeMcp()


@pytest.fixture
def deps(fake_mcp: FakeMcp) -> RuntimeDeps:
    deps = RuntimeDeps.lazy()
    deps.use(mcp=fake_mcp)
    return deps


def _node(deps: RuntimeDeps, **params: Any) -> Any:
    return ocr_extract(name="ocr", deps=deps, server="ocr", tool="extract_text", **params)


async def test_ok_route_outputs_the_server_contract_and_fills_state(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", answer())

    run = await run_node(_node(deps), {"image_path": "order.png"}, {"run.id": RUN_ID})

    assert run.route == "ok"
    result = OcrResult.model_validate(run.output)
    assert result.text == MASKED
    [(server, tool, args)] = fake_mcp.calls
    assert (server, tool) == ("ocr", "extract_text")
    assert args == {"image_base64": base64.b64encode(PNG).decode(), "request_id": RUN_ID}
    assert run.state["run.document_sha256"] == hashlib.sha256(PNG).hexdigest()
    assert run.state["run.patient_ref"] == PATIENT_REF
    assert run.state["run.pii_counts"] == {"CPF": 1, "PERSON": 1}


async def test_image_never_reaches_output_or_state(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", answer())

    run = await run_node(_node(deps), {"image_path": "order.png"}, {"run.id": RUN_ID})

    encoded = base64.b64encode(PNG).decode()
    dumped = json.dumps([run.state, [e.model_dump(mode="json") for e in run.events]])
    assert encoded not in dumped
    assert "image_base64" not in dumped


@pytest.mark.parametrize(
    ("kwargs", "params"),
    [
        ({"confidence": 0.5}, {}),
        ({"confidence": 0.7}, {"min_mean_confidence": 0.8}),
        ({"text": ""}, {}),
        ({"text": "   "}, {}),
    ],
)
async def test_low_quality_route(
    root: Path,
    deps: RuntimeDeps,
    fake_mcp: FakeMcp,
    kwargs: dict[str, Any],
    params: dict[str, Any],
) -> None:
    fake_mcp.script("ocr", "extract_text", answer(**kwargs))

    run = await run_node(_node(deps, **params), {"image_path": "order.png"})

    assert run.route == "low_quality"
    assert RunFailure.model_validate(run.output).code == "OCR_LOW_QUALITY"
    assert "run.document_sha256" in run.state


async def test_missing_run_id_still_sends_a_uuid(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", answer())

    await run_node(_node(deps), {"image_path": str(root / "order.png")})

    request_id = fake_mcp.calls[0][2]["request_id"]
    assert len(request_id) == 36


async def test_result_for_another_document_is_permanent(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", ocr_body(base64.b64encode(b"other").decode()))

    with pytest.raises(PermanentError, match="OCR_DOCUMENT_MISMATCH"):
        await run_node(_node(deps), {"image_path": "order.png"})


async def test_result_breaking_the_contract_is_permanent_without_its_text(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", {"text": "CPF 529.982.247-25"})

    with pytest.raises(PermanentError) as caught:
        await run_node(_node(deps), {"image_path": "order.png"})

    assert caught.value.code == "OCR_INVALID_RESULT"
    assert "529" not in str(caught.value)


async def test_input_tool_errors_become_invalid_input(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    error = PermanentToolError("MCP_TOOL_ERROR")
    error.tool_code, error.retryable = "CORRUPTED_IMAGE", False
    fake_mcp.script("ocr", "extract_text", error)

    with pytest.raises(InvalidInputError, match="CORRUPTED_IMAGE"):
        await run_node(_node(deps), {"image_path": "order.png"})


async def test_other_tool_errors_propagate(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    error = PermanentToolError("MCP_TOOL_ERROR")
    error.tool_code, error.retryable = "PII_ENGINE_FAILURE", False
    fake_mcp.script("ocr", "extract_text", error)

    with pytest.raises(PermanentToolError):
        await run_node(_node(deps), {"image_path": "order.png"})


async def test_transient_failures_are_retried_by_the_node(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    busy = TransientToolError("MCP_TOOL_ERROR")
    busy.tool_code, busy.retryable = "BUSY", True
    fake_mcp.script("ocr", "extract_text", busy, TransientError("MCP_TIMEOUT"), answer())
    retry = RetryConfig(max_attempts=3, initial_delay=0.0, max_delay=0.0, jitter=0.0)

    run = await run_node(_node(deps, retry=retry), {"image_path": "order.png"})

    assert run.route == "ok"
    assert len(fake_mcp.calls) == 3


async def test_permanent_failures_are_not_retried(
    root: Path, deps: RuntimeDeps, fake_mcp: FakeMcp
) -> None:
    fake_mcp.script("ocr", "extract_text", PermanentError("MCP_UNAUTHORIZED"))
    retry = RetryConfig(max_attempts=3, initial_delay=0.0, max_delay=0.0, jitter=0.0)

    with pytest.raises(PermanentError):
        await run_node(_node(deps, retry=retry), {"image_path": "order.png"})

    assert len(fake_mcp.calls) == 1


def test_load_image_accepts_png_and_jpeg(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(JPEG)

    assert load_image("a.jpg", tmp_path.resolve(), 1024) == JPEG
    assert load_image(str(tmp_path / "a.jpg"), tmp_path.resolve(), 1024) == JPEG


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("missing.png", None, "IMAGE_NOT_FOUND"),
        ("empty.png", b"", "IMAGE_EMPTY"),
        ("big.png", PNG + b"\x00" * 2048, "IMAGE_TOO_LARGE"),
        ("text.png", b"hello world", "UNSUPPORTED_FORMAT"),
        ("gif.png", b"GIF89a" + b"\x00" * 10, "UNSUPPORTED_FORMAT"),
    ],
)
def test_load_image_rejects(tmp_path: Path, name: str, content: bytes | None, code: str) -> None:
    if content is not None:
        (tmp_path / name).write_bytes(content)

    with pytest.raises(InvalidInputError) as caught:
        load_image(name, tmp_path.resolve(), 1024)

    assert caught.value.code == code
    assert name not in str(caught.value)


def test_load_image_stays_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.png").write_bytes(PNG)
    (root / "link.png").symlink_to(tmp_path / "outside.png")
    (root / "dir.png").mkdir()

    for path in ("../outside.png", str(tmp_path / "outside.png"), "link.png"):
        with pytest.raises(InvalidInputError, match="IMAGE_OUTSIDE_INPUT_ROOT"):
            load_image(path, root.resolve(), 1024)
    with pytest.raises(InvalidInputError, match="IMAGE_NOT_A_FILE"):
        load_image("dir.png", root.resolve(), 1024)


@pytest.mark.parametrize(
    "value",
    [
        types.Content(role="user", parts=[types.Part(text='{"image_path": "a.png"}')]),
        '{"image_path": "a.png"}',
        {"image_path": "a.png"},
        RunInput(image_path="a.png"),
    ],
)
def test_parse_run_input(value: object) -> None:
    assert parse_run_input(value) == RunInput(image_path="a.png")


@pytest.mark.parametrize("value", ["not json", {"image": "a"}, None, 3, '{"image_path": 1}'])
def test_parse_run_input_rejects(value: object) -> None:
    with pytest.raises(InvalidInputError, match="INVALID_RUN_INPUT"):
        parse_run_input(value)


def test_params_are_validated() -> None:
    deps = RuntimeDeps.lazy()
    with pytest.raises(ValueError, match="min_mean_confidence"):
        _node(deps, min_mean_confidence=1.5)
    with pytest.raises(ValueError, match="max_image_bytes"):
        _node(deps, max_image_bytes=0)
