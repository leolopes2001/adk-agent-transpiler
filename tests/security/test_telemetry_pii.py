"""No personal data, prompt, model answer, image or document text reaches the telemetry.

The real stack processes the same orders as test_runtime_pii (real OCR, RAG,
API and Postgres; FakeLlm as the model), with the exporters of
`lab_common.telemetry` writing to memory instead of a collector. Every span
(name, attributes, events, status) and every exported log record (body and
attributes) is then searched for the samples' personal data, the images, the
prompt the model received, the model's answers, the masked document text and
the patient's pseudonym.

The last tests run the real `lab-agent` CLI in a subprocess against a fake
OTLP collector: without OTEL_EXPORTER_OTLP_ENDPOINT it receives nothing; with
it, it receives traces and logs, and the raw OTLP bytes are checked too.
"""

import asyncio
import base64
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from pii_guard import high_precision_detectors
from pii_guard.redactor import detect_all
from runtime_stack import FAKE_LLM_FIXTURES, REPO_ROOT, SAMPLE_PII, SPEC, AgentRun, Stack, run_agent
from telemetry_capture import Exported

pytestmark = pytest.mark.security

UNCAPITALIZED = "nomes_sem_capitalizacao.png"
NO_LABEL = "nomes_sem_rotulo.png"
RUNS = [
    ("samples/pedido_medico_ficticio.png", "pedido_medico_ficticio.json"),
    ("samples/pedido_ambiguo.png", "pedido_ambiguo.json"),
    ("samples/pedido_injecao.png", "pedido_medico_ficticio.json"),
    (UNCAPITALIZED, "pedido_medico_ficticio.json"),
    (NO_LABEL, "pedido_medico_ficticio.json"),
]
MIN_CHUNK = 12
"""Prompt and document lines are searched in pieces of at least this many characters."""


@dataclass
class TelemetryRun:
    run: AgentRun
    spans: str
    logs: str
    span_names: set[str]


def _span_text(exported: Exported) -> tuple[str, set[str]]:
    parts = []
    for span in exported.spans():
        parts.append(span.name)
        parts.append(json.dumps(dict(span.attributes or {}), ensure_ascii=False, default=str))
        parts.append(str(span.status.description))
        for event in span.events:
            parts.append(event.name)
            parts.append(json.dumps(dict(event.attributes or {}), ensure_ascii=False, default=str))
    return "\n".join(parts), {span.name for span in exported.spans()}


def _log_text(exported: Exported) -> str:
    parts = []
    for record in exported.logs():
        parts.append(str(record.log_record.body))
        parts.append(json.dumps(dict(record.log_record.attributes or {}), ensure_ascii=False))
    return "\n".join(parts)


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def runs(
    stack: Stack, _exported_session: Exported, monkeypatch_module: pytest.MonkeyPatch
) -> list[TelemetryRun]:
    for image in (UNCAPITALIZED, NO_LABEL):
        shutil.copy(REPO_ROOT / "tests" / "data" / "images" / image, stack.work)
    exported = _exported_session
    results = []
    for image, fixture in RUNS:
        exported.clear()
        run = asyncio.run(
            run_agent(
                stack,
                monkeypatch_module,
                image,
                fixture=fixture,
                log_export=exported.telemetry.log_export,
            )
        )
        spans, names = _span_text(exported)
        results.append(TelemetryRun(run, spans, _log_text(exported), names))
    exported.clear()
    return results


def _surface(runs: list[TelemetryRun], surface: str) -> str:
    return "\n".join(getattr(r, surface) for r in runs)


def _chunks(text: str) -> set[str]:
    """Lines of `text`, and the parts of each line between punctuation, long enough to search."""
    pieces = set()
    for line in text.splitlines():
        for piece in [line, *line.replace(":", ".").split(".")]:
            if len(piece.strip()) >= MIN_CHUNK:
                pieces.add(piece.strip())
    return pieces


def test_every_service_and_the_model_call_are_traced(runs: list[TelemetryRun]) -> None:
    names = runs[0].span_names

    assert {"lab-agent run", "invoke_node ocr", "invoke_agent exam_extractor"} <= names
    assert {"tools/call extract_text", "tools/call batch_search"} <= names
    assert {"mcp ocr.extract_text", "mcp rag.batch_search", "POST /v1/appointments"} <= names
    assert any(name.startswith("generate_content") for name in names)
    assert runs[0].logs, "no log record was exported"


@pytest.mark.parametrize("surface", ["spans", "logs"])
def test_no_personal_data(runs: list[TelemetryRun], surface: str) -> None:
    text = _surface(runs, surface)

    assert [value for value in SAMPLE_PII if value in text] == []
    assert [span.entity.value for span in detect_all(high_precision_detectors(), text, 0.0)] == []


@pytest.mark.parametrize("surface", ["spans", "logs"])
def test_no_image(runs: list[TelemetryRun], stack: Stack, surface: str) -> None:
    text = _surface(runs, surface)

    for image, _ in RUNS:
        encoded = base64.b64encode((stack.work / image).read_bytes()).decode()
        assert encoded[:64] not in text
        assert encoded[-64:] not in text
    assert "image_base64" not in text


@pytest.mark.parametrize("surface", ["spans", "logs"])
def test_no_prompt(runs: list[TelemetryRun], surface: str) -> None:
    text = _surface(runs, surface)
    prompts = set()
    for run in runs:
        for request in run.run.llm_requests:
            if request.config is not None:
                prompts |= _chunks(str(request.config.system_instruction))
            for content in request.contents:
                prompts |= _chunks("".join(part.text or "" for part in content.parts or []))

    assert prompts, "the model received nothing: nothing was checked"
    assert [piece for piece in prompts if piece in text] == []


@pytest.mark.parametrize("surface", ["spans", "logs"])
def test_no_model_answer(runs: list[TelemetryRun], surface: str) -> None:
    text = _surface(runs, surface)
    answers = set()
    for _, fixture in RUNS:
        for exam in json.loads((FAKE_LLM_FIXTURES / fixture).read_text("utf-8"))["exams"]:
            answers |= {exam["name"], exam["evidence"]}

    assert [answer for answer in answers if len(answer) > 3 and answer in text] == []
    # The answer's shape: exam codes (allowed) are logged under "exams", evidence never.
    assert '"evidence"' not in text


@pytest.mark.parametrize("surface", ["spans", "logs"])
def test_no_document_text_or_patient_ref(runs: list[TelemetryRun], surface: str) -> None:
    text = _surface(runs, surface)
    document: set[str] = set()
    patient_refs = []
    for run in runs:
        for session in run.run.sessions:
            document |= _chunks(str(session.state.get("doc.text") or ""))
            if session.state.get("run.patient_ref"):
                patient_refs.append(str(session.state["run.patient_ref"]))

    assert document, "no document text in the sessions: nothing was checked"
    assert patient_refs
    assert [line for line in document if line in text] == []
    assert [ref for ref in patient_refs if ref in text] == []
    assert "[PESSOA_" not in text


class _Collector(http.server.BaseHTTPRequestHandler):
    received: list[tuple[str, bytes]]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.received.append((self.path, self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def collector() -> Iterator[tuple[str, list[tuple[str, bytes]]]]:
    """A fake OTLP/HTTP collector on a free loopback port: (endpoint, requests received)."""
    received: list[tuple[str, bytes]] = []
    handler = type("Collector", (_Collector,), {"received": received})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", received
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _cli(stack: Stack, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """`lab-agent run` in a new process against the in-process stack."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    environment.update(stack.env)
    environment.update(env)
    return subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
        [
            sys.executable,
            "-m",
            "agent_runtime.cli",
            "run",
            "--image",
            "samples/pedido_medico_ficticio.png",
            "--agent-dir",
            str(stack.agent_dir),
            "--spec",
            str(SPEC),
            "--non-interactive",
            "--output",
            "json",
        ],
        env=environment,
        cwd=stack.work,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_nothing_is_exported_without_the_endpoint(
    stack: Stack, collector: tuple[str, list[tuple[str, bytes]]]
) -> None:
    _, received = collector

    result = _cli(stack, {"LOG_LEVEL": ""})

    assert result.returncode == 0, result.stderr
    assert received == []
    # The terminal is as quiet as always: no log line in the default (ERROR) level.
    assert '"level": "info"' not in result.stderr


def test_with_the_endpoint_only_safe_bytes_reach_the_collector(
    stack: Stack, collector: tuple[str, list[tuple[str, bytes]]]
) -> None:
    endpoint, received = collector

    result = _cli(stack, {"LOG_LEVEL": "", "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint})

    assert result.returncode == 0, result.stderr
    assert '"level": "info"' not in result.stderr
    assert {path for path, _ in received} == {"/v1/traces", "/v1/logs"}
    payload = b"\n".join(body for _, body in received)
    assert b"lab-agent run" in payload
    assert b"run_finished" in payload
    image = (stack.work / "samples" / "pedido_medico_ficticio.png").read_bytes()
    encoded = base64.b64encode(image)
    forbidden = [*(v.encode() for v in SAMPLE_PII), encoded[:64], encoded[-64:]]
    forbidden += [b"Hemograma completo", b"[PESSOA_1]", b"image_base64"]
    assert [value for value in forbidden if value in payload] == []


def test_without_a_collector_listening_nothing_is_attempted(stack: Stack) -> None:
    # What every service sees without the `observability` profile: the name does not resolve.
    result = _cli(
        stack,
        {"LOG_LEVEL": "", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://observability.invalid:18890"},
    )

    assert result.returncode == 0, result.stderr
    assert "opentelemetry" not in result.stderr.lower()
    assert "Failed to export" not in result.stderr
