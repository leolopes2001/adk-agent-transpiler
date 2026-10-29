"""No personal data leaves the OCR server through the runtime.

The real stack processes the three samples and two synthetic orders (real OCR,
RAG, API and Postgres; FakeLlm as the model), one of them with a human review.
Then every place the runtime writes to is searched for the personal data
printed on the images and for the image itself: the requests the model
received, the bodies sent to and from the MCP servers and the API, the logs of
the whole process, the ADK session (events and state), the report, the CLI
output, the review screen and the scheduler database.
"""

import asyncio
import base64
import io
import json
import re
import shutil
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

from agent_runtime.guardrails.document import split_document
from agent_runtime.terminal import review_intro, review_item
from pii_guard import high_precision_detectors
from pii_guard.redactor import detect_all
from runtime_stack import REPO_ROOT, SAMPLE_PII, AgentRun, Stack, run_agent

pytestmark = pytest.mark.security

PII_VALUES = SAMPLE_PII
UNCAPITALIZED = "nomes_sem_capitalizacao.png"
NO_LABEL = "nomes_sem_rotulo.png"
RUNS = [
    ("samples/pedido_medico_ficticio.png", "pedido_medico_ficticio.json"),
    ("samples/pedido_ambiguo.png", "pedido_ambiguo.json"),
    ("samples/pedido_injecao.png", "pedido_medico_ficticio.json"),
    (UNCAPITALIZED, "pedido_medico_ficticio.json"),
    (NO_LABEL, "pedido_medico_ficticio.json"),
]
REVIEWED = ("samples/pedido_ambiguo.png", "pedido_ambiguo.json")
SURFACES = [
    "llm_requests",
    "ocr_responses",
    "rag_traffic",
    "api_traffic",
    "logs",
    "session",
    "report",
    "cli_output",
    "review_screen",
    "database",
]


class ScreenReview:
    """Shows the review as `lab-agent run` does, keeps the screen and picks the first option."""

    def __init__(self) -> None:
        self.screens: list[str] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        console = Console(record=True, width=120, file=io.StringIO())
        items = list(payload.get("items", []))
        console.print(review_intro(len(items)))
        for position, item in enumerate(items, start=1):
            console.print(review_item(item, position, len(items)))
        self.screens.append(console.export_text())
        return {"choices": {str(item["index"]): item["options"][0]["code"] for item in items}}


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def review() -> ScreenReview:
    return ScreenReview()


@pytest.fixture(scope="module")
def runs(
    stack: Stack, monkeypatch_module: pytest.MonkeyPatch, review: ScreenReview
) -> list[AgentRun]:
    for image in (UNCAPITALIZED, NO_LABEL):
        shutil.copy(REPO_ROOT / "tests" / "data" / "images" / image, stack.work)
    done = [
        asyncio.run(run_agent(stack, monkeypatch_module, image, fixture=fixture))
        for image, fixture in RUNS
    ]
    image, fixture = REVIEWED
    done.append(
        asyncio.run(
            run_agent(
                stack, monkeypatch_module, image, fixture=fixture, interactive=True, review=review
            )
        )
    )
    return done


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


def _documents(run: AgentRun) -> str:
    """The document blocks the model received, without the fixed instruction."""
    blocks: list[str] = []
    for request in run.llm_requests:
        for content in request.contents:
            for part in content.parts or []:
                blocks.extend(split_document(part.text or "")[0])
    return "\n".join(blocks)


def _surfaces(runs: list[AgentRun], stack: Stack, review: ScreenReview) -> dict[str, str]:
    return {
        "llm_requests": "\n".join(_llm_text(r) for r in runs),
        # The image goes to the OCR server; what comes back is masked text.
        "ocr_responses": stack.wire["ocr"].text(received=False),
        "rag_traffic": stack.wire["rag"].text(),
        "api_traffic": stack.wire["api"].text(),
        "logs": "\n".join(r.logs for r in runs),
        "session": "\n".join(_session_text(r) for r in runs),
        "report": "\n".join(json.dumps(r.report, ensure_ascii=False) for r in runs),
        "cli_output": "\n".join(r.output for r in runs),
        "review_screen": "\n".join(review.screens),
        "database": stack.dump_database(),
    }


@pytest.mark.parametrize("surface", SURFACES)
def test_no_personal_data_on_any_surface(
    runs: list[AgentRun], stack: Stack, review: ScreenReview, surface: str
) -> None:
    text = _surfaces(runs, stack, review)[surface]

    assert text, f"{surface} is empty: nothing was checked"
    leaked = [value for value in PII_VALUES if value in text]
    assert leaked == []
    found = detect_all(high_precision_detectors(), text, 0.0)
    assert [span.entity.value for span in found] == []


@pytest.mark.parametrize("surface", SURFACES)
def test_the_image_never_leaves_the_ocr_node(
    runs: list[AgentRun], stack: Stack, review: ScreenReview, surface: str
) -> None:
    text = _surfaces(runs, stack, review)[surface]

    for image, _ in [*RUNS, REVIEWED]:
        encoded = base64.b64encode((stack.work / image).read_bytes()).decode()
        assert encoded[:64] not in text
        assert encoded[-64:] not in text
    # The OCR server's tool list names the argument; only its value must not come back.
    if surface != "ocr_responses":
        assert "image_base64" not in text


def test_the_model_received_only_exam_lines_inside_the_delimiters(runs: list[AgentRun]) -> None:
    document = _documents(runs[0])

    assert "<documento_nao_confiavel>" in _llm_text(runs[0])
    assert "Hemograma completo" in document
    assert "Exames solicitados:" in document
    # Minimization: no personal-data field reaches the model, not even masked.
    for absent in ("[PESSOA_", "[CPF_", "Paciente", "CPF", "Convênio", "Dr.", "Belo Horizonte"):
        assert absent not in document


def test_patient_is_only_a_pseudonym(runs: list[AgentRun], stack: Stack) -> None:
    states: list[dict[str, Any]] = [s.state for s in runs[0].sessions]

    [state] = states
    assert str(state["run.patient_ref"]).startswith("pt_")
    assert state["run.patient_ref"] in stack.dump_database()
    assert "doc.text" in state
    assert "[PESSOA_1]" in state["doc.text"]


def test_labeled_names_in_any_case_are_masked_and_kept_from_the_model(
    runs: list[AgentRun],
) -> None:
    run = runs[3]
    [state] = [s.state for s in run.sessions]
    document = _documents(run)

    placeholders = set(re.findall(r"\[PESSOA_\d+\]", state["doc.text"]))
    assert {"[PESSOA_1]", "[PESSOA_2]", "[PESSOA_3]"} <= placeholders
    assert "[PESSOA_" not in document
    assert "Hemograma completo" in document
    assert "T4 livre" in document
    assert run.report["pii_masked"]["PERSON"] == 3


def test_names_without_a_label_are_masked_by_the_ner_model(runs: list[AgentRun]) -> None:
    run = runs[4]
    [state] = [s.state for s in run.sessions]

    assert run.report["pii_masked"] == {"PERSON": 3}
    assert len(set(re.findall(r"\[PESSOA_\d+\]", state["doc.text"]))) == 3
    assert "3. Ferritina" in _documents(run)


def test_the_review_screen_was_shown(review: ScreenReview, runs: list[AgentRun]) -> None:
    assert runs[-1].report["outcome"] in {"scheduled", "replayed"}
    assert review.screens
    assert "Lido no pedido" in review.screens[0]
