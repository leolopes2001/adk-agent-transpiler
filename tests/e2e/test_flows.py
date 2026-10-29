"""The flows of `lab-agent run` in the Compose stack: image -> OCR MCP -> guard ->
extraction -> RAG MCP -> gate -> scheduler API -> CLI, plus the failure routes.

Every run is `docker compose run --rm agent ...` with the generated code from
the volume; the assertions use the CLI's JSON report, the API and the database.
"""

import json
import re
import uuid
from typing import Any

import pytest

from e2e.helpers import (
    FAKE_LLM_FIXTURES,
    MAIN_SPEC,
    REPO_ROOT,
    SAMPLES,
    TEST_IMAGES,
    AgentRun,
    ComposeStack,
)
from lab_common.contracts.ocr import PATIENT_REF_PATTERN

pytestmark = pytest.mark.e2e

EXPECTED = json.loads(
    (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
)
EXPECTED_CODES = [exam["code"] for exam in EXPECTED["exams"]]
HAPPY_IMAGE = f"{SAMPLES}/pedido_medico_ficticio.png"
JSON_RUN = ("--output", "json", "--non-interactive")
LLM_QUOTA_ERRORS = {"LLM_RESOURCE_EXHAUSTED", "LLM_429", "LLM_402"}
HANDWRITTEN_IMAGE = f"{SAMPLES}/pedido_manuscrito.jpg"
# Hemograma completo, Glicemia de jejum, TSH and Ferritina: the exams written on the photo.
HANDWRITTEN_CODES = {"LAB-10001", "LAB-10012", "LAB-10043", "LAB-10109"}
# The patient's data on the photo (fictional), whole and in pieces.
HANDWRITTEN_PII = [
    "João Paulo da Silva",
    "JOÃO",
    "SILVA",
    "123.456.789-09",
    "12345678909",
    "123.456",
    "456.789",
    "789-09",
    "(11) 90000-0000",
    "(11)",
    "90000-0000",
    "90000",
    "01/01/1990",
    "01/01",
]


def _accepted_codes(run: AgentRun) -> list[str]:
    assert run.report is not None, run.stderr
    return [e["code"] for e in run.report["exams"] if e["status"] == "accepted"]


def _api_events(stack: ComposeStack) -> list[dict[str, Any]]:
    events = []
    for line in stack.logs("--no-log-prefix", "scheduler-api").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _api_requests_of(stack: ComposeStack, run_id: str) -> list[dict[str, Any]]:
    return [
        e
        for e in _api_events(stack)
        if e.get("event") == "request" and e.get("request_id") == run_id
    ]


@pytest.fixture(scope="module")
def happy_run(e2e_stack: ComposeStack) -> AgentRun:
    return e2e_stack.run_agent(HAPPY_IMAGE, *JSON_RUN, "--run-id", str(uuid.uuid4()))


def test_happy_path_schedules_the_expected_exams(
    e2e_stack: ComposeStack, happy_run: AgentRun
) -> None:
    report = happy_run.report

    assert happy_run.exit_code == 0, happy_run.stderr
    assert report is not None
    assert report["outcome"] == "scheduled"
    assert report["appointment"]["status"] == "scheduled"
    assert _accepted_codes(happy_run) == EXPECTED_CODES
    assert report["path"] == [
        "ocr",
        "guard",
        "extraction",
        "resolve",
        "gate",
        "schedule",
        "present",
    ]

    status, appointment = e2e_stack.api_get(f"/v1/appointments/{report['appointment']['id']}")

    assert status == 200
    assert sorted(e["code"] for e in appointment["exams"]) == sorted(EXPECTED_CODES)
    assert re.fullmatch(PATIENT_REF_PATTERN, appointment["patient_ref"])
    # The run id travels as X-Request-ID: the API's log of the POST carries it.
    posts = [e for e in _api_requests_of(e2e_stack, report["run_id"]) if e["method"] == "POST"]
    assert [e["status"] for e in posts] == [201]


def test_second_run_is_an_idempotent_replay(e2e_stack: ComposeStack, happy_run: AgentRun) -> None:
    assert happy_run.report is not None
    first_id = happy_run.report["appointment"]["id"]
    before = e2e_stack.count("appointments")

    replay = e2e_stack.run_agent(HAPPY_IMAGE, *JSON_RUN)

    assert replay.exit_code == 0, replay.stderr
    assert replay.report is not None
    assert replay.report["outcome"] == "replayed"
    assert replay.report["appointment"] == {"id": first_id, "status": "replayed"}
    assert e2e_stack.count("appointments") == before
    documents = dict(e2e_stack.db_rows("SELECT id, source_document_sha256 FROM appointments"))
    assert list(documents.values()).count(documents[first_id]) == 1


def test_injection_is_blocked_before_any_api_call(e2e_stack: ComposeStack) -> None:
    run_id = str(uuid.uuid4())
    before = e2e_stack.count("appointments")

    run = e2e_stack.run_agent(f"{SAMPLES}/pedido_injecao.png", *JSON_RUN, "--run-id", run_id)

    assert run.exit_code == 3, run.stderr
    assert run.report is not None
    assert run.report["outcome"] == "INPUT_BLOCKED"
    assert run.report["path"] == ["ocr", "guard", "abort"]
    assert run.report["injection_severity"] == "high"
    assert _api_requests_of(e2e_stack, run_id) == []
    assert e2e_stack.count("appointments") == before


def test_ambiguous_item_is_rejected_when_non_interactive(e2e_stack: ComposeStack) -> None:
    run = e2e_stack.run_agent(
        f"{SAMPLES}/pedido_ambiguo.png",
        *JSON_RUN,
        env={"FAKE_LLM_FIXTURE": f"{FAKE_LLM_FIXTURES}/pedido_ambiguo.json"},
    )

    assert run.exit_code == 0, run.stderr
    assert run.report is not None
    exams = {e["mention"]: e for e in run.report["exams"]}
    assert (exams["T4"]["status"], exams["T4"]["reason"]) == ("rejected", "needs_human_review")
    assert (exams["Hemograma completo"]["status"], exams["Hemograma completo"]["code"]) == (
        "accepted",
        "LAB-10001",
    )
    assert run.report["appointment"]["status"] in {"scheduled", "replayed"}

    status, appointment = e2e_stack.api_get(f"/v1/appointments/{run.report['appointment']['id']}")

    assert status == 200
    assert [e["code"] for e in appointment["exams"]] == ["LAB-10001"]


def test_illegible_image_ends_with_low_quality(e2e_stack: ComposeStack) -> None:
    run = e2e_stack.run_agent(f"{TEST_IMAGES}/blank.png", *JSON_RUN)

    assert run.exit_code == 4, run.stderr
    assert run.report is not None
    assert run.report["outcome"] == "OCR_LOW_QUALITY"
    assert run.report["path"] == ["ocr", "abort"]


def test_handwritten_order_on_ruled_paper_is_read_without_personal_data(
    e2e_stack: ComposeStack,
) -> None:
    runs = [
        e2e_stack.run_agent(
            HANDWRITTEN_IMAGE,
            "--non-interactive",
            *output,
            env={"FAKE_LLM_FIXTURE": f"{FAKE_LLM_FIXTURES}/pedido_manuscrito.json"},
        )
        for output in (("--output", "json"), ("--output", "table"))
    ]
    report = runs[0].report

    assert [run.exit_code for run in runs] == [0, 0], runs[0].stderr
    assert report is not None
    assert report["path"][:3] == ["ocr", "guard", "extraction"]
    accepted = set(_accepted_codes(runs[0]))
    assert accepted
    # The fixture also names T4 livre, which the photo does not have.
    assert accepted <= HANDWRITTEN_CODES
    output = "".join(run.stdout + run.stderr for run in runs) + e2e_stack.logs()
    assert [value for value in HANDWRITTEN_PII if value in output] == []


def test_alternative_spec_looks_exams_up_without_scheduling(e2e_stack: ComposeStack) -> None:
    run_id = str(uuid.uuid4())
    before = e2e_stack.count("appointments")

    run = e2e_stack.run_agent(
        HAPPY_IMAGE,
        "--agent", "exam_lookup_only",
        "--non-interactive",
        "--run-id", run_id,
    )  # fmt: skip

    assert run.exit_code == 0, run.stderr
    # The spec's own output format: a table, with the codes the catalog resolved.
    assert re.findall(r"LAB-\d{5}", run.stdout)
    assert set(re.findall(r"LAB-\d{5}", run.stdout)) <= set(EXPECTED_CODES)
    assert "este agente não registra solicitação" in run.stdout
    assert "Exames identificados (" in run.stdout
    assert f"Execução {run_id}" in run.stdout
    assert _api_requests_of(e2e_stack, run_id) == []
    assert e2e_stack.count("appointments") == before


def test_list_names_the_generated_agents(e2e_stack: ComposeStack) -> None:
    result = e2e_stack.compose("run", "--rm", "-T", "agent", "list", check=False)

    assert result.returncode == 0, result.stderr
    names = [line.split()[0] for line in result.stdout.splitlines()]
    assert {"exam_scheduling_agent", "exam_lookup_only"} <= set(names)
    assert "desatualizado" not in result.stdout


def test_spec_changed_without_regenerating_is_refused(e2e_stack: ComposeStack) -> None:
    # The mounted spec is read-only: a copy with a different threshold goes to /tmp.
    script = (
        "import json, sys\n"
        "spec = json.load(open(sys.argv[1]))\n"
        "for node in spec['workflow']['nodes']:\n"
        "    if node['kind'] == 'confidence_gate':\n"
        "        node['params']['accept_threshold'] = 0.95\n"
        "json.dump(spec, open('/tmp/spec.json', 'w'))\n"
    )
    shell = (
        f'python -c "$0" {MAIN_SPEC} && exec lab-agent run --image {HAPPY_IMAGE} '
        "--spec /tmp/spec.json --output json --non-interactive"
    )

    result = e2e_stack.compose(
        "run", "--rm", "-T", "--entrypoint", "sh", "agent", "-c", shell, script, check=False
    )

    assert result.returncode == 7, result.stderr
    assert "o código gerado está desatualizado em relação à spec" in result.stderr
    assert result.stdout == ""


@pytest.mark.llm
def test_real_gemini_schedules_the_expected_exams(e2e_stack: ComposeStack, llm_key: str) -> None:
    run = e2e_stack.run_agent(
        HAPPY_IMAGE,
        *JSON_RUN,
        env={"APP_ENV": "dev", "AGENT_LLM_BACKEND": "gemini"},
        pass_env=(llm_key,),
    )

    assert run.report is not None, run.stderr
    if run.report.get("error_code") in LLM_QUOTA_ERRORS:
        pytest.skip(f"Gemini quota: {run.report['error_code']}")
    assert run.exit_code == 0, run.report.get("error_code")
    # The model may name the exams differently, and an extra mention may go to review
    # (rejected when non-interactive); the accepted and scheduled codes may not differ:
    # no code missing, none extra, none twice.
    assert sorted(_accepted_codes(run)) == sorted(EXPECTED_CODES)
    assert run.report["appointment"]["status"] in {"scheduled", "replayed"}

    status, appointment = e2e_stack.api_get(f"/v1/appointments/{run.report['appointment']['id']}")

    assert status == 200
    assert sorted(e["code"] for e in appointment["exams"]) == sorted(EXPECTED_CODES)
