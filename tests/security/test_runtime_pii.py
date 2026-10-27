"""No personal data leaves the OCR server through the runtime.

The real stack processes the three samples (real OCR, RAG, API and Postgres;
FakeLlm as the model). Then every place the runtime writes to is searched for
the personal data printed on the samples and for the image itself: the
requests the model received, the logs of the whole process, the ADK session
(events and state), the report and the scheduler database.
"""

import asyncio
import base64
import json
from collections.abc import Iterator
from typing import Any

import pytest

from lab_common.pii import high_precision_detectors
from lab_common.pii.redactor import detect_all
from runtime_stack import REPO_ROOT, AgentRun, Stack, run_agent

pytestmark = pytest.mark.security

# Printed on the samples by scripts/make_sample_images.py (all fictional).
PII_VALUES = [
    "Maria",
    "MARIA",
    "Fictícia da Silva",
    "202.393.613-69",
    "20239361369",
    "14/03/1985",
    "(31) 90000-0000",
    "90000-0000",
    "0000 1111 2222 3333",
    "João Exemplo",
    "CRM-MG 000000",
    "Rua Imaginária",
]
RUNS = [
    ("samples/pedido_medico_ficticio.png", "pedido_medico_ficticio.json"),
    ("samples/pedido_ambiguo.png", "pedido_ambiguo.json"),
    ("samples/pedido_injecao.png", "pedido_medico_ficticio.json"),
]


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def runs(stack: Stack, monkeypatch_module: pytest.MonkeyPatch) -> list[AgentRun]:
    return [
        asyncio.run(run_agent(stack, monkeypatch_module, image, fixture=fixture))
        for image, fixture in RUNS
    ]


def _llm_text(run: AgentRun) -> str:
    parts: list[str] = []
    for request in run.llm_requests:
        parts.extend(c.model_dump_json() for c in request.contents)
        if request.config is not None:
            parts.append(str(request.config.system_instruction))
    return "\n".join(parts)


def _session_text(run: AgentRun) -> str:
    return "\n".join(
        json.dumps(session.state, ensure_ascii=False, default=str)
        + "\n"
        + "\n".join(e.model_dump_json() for e in session.events)
        for session in run.sessions
    )


def _surfaces(runs: list[AgentRun], stack: Stack) -> dict[str, str]:
    return {
        "llm_requests": "\n".join(_llm_text(r) for r in runs),
        "logs": "\n".join(r.logs for r in runs),
        "session": "\n".join(_session_text(r) for r in runs),
        "report": "\n".join(json.dumps(r.report, ensure_ascii=False) for r in runs),
        "database": stack.dump_database(),
    }


@pytest.mark.parametrize("surface", ["llm_requests", "logs", "session", "report", "database"])
def test_no_personal_data_on_any_surface(runs: list[AgentRun], stack: Stack, surface: str) -> None:
    text = _surfaces(runs, stack)[surface]

    assert text, f"{surface} is empty: nothing was checked"
    leaked = [value for value in PII_VALUES if value in text]
    assert leaked == []
    found = detect_all(high_precision_detectors(), text, 0.0)
    assert [span.entity.value for span in found] == []


@pytest.mark.parametrize("surface", ["llm_requests", "logs", "session", "report", "database"])
def test_the_image_never_leaves_the_ocr_node(
    runs: list[AgentRun], stack: Stack, surface: str
) -> None:
    text = _surfaces(runs, stack)[surface]

    for image, _ in RUNS:
        encoded = base64.b64encode((REPO_ROOT / image).read_bytes()).decode()
        assert encoded[:64] not in text
        assert encoded[-64:] not in text
    assert "image_base64" not in text


def test_the_model_received_masked_text_inside_the_delimiters(runs: list[AgentRun]) -> None:
    text = _llm_text(runs[0])

    assert "[PESSOA_1]" in text
    assert "[CPF_1]" in text
    assert "<documento_nao_confiavel>" in text
    assert "Hemograma completo" in text


def test_patient_is_only_a_pseudonym(runs: list[AgentRun], stack: Stack) -> None:
    states: list[dict[str, Any]] = [s.state for s in runs[0].sessions]

    [state] = states
    assert str(state["run.patient_ref"]).startswith("pt_")
    assert state["run.patient_ref"] in stack.dump_database()
    assert "doc.text" in state
    assert "[PESSOA_1]" in state["doc.text"]
