"""No personal data in anything the Compose stack writes.

The three samples run through the containers (FakeLlm as the model). Then the
logs of every container, the agent's output and the database are searched for
the personal data printed on the samples, with and without masks, and for the
generic high-precision patterns (CPF with valid check digits, CNS, e-mail,
phone).
"""

import base64

import pytest

from e2e.helpers import FAKE_LLM_FIXTURES, REPO_ROOT, SAMPLES, AgentRun, ComposeStack
from lab_common.pii import high_precision_detectors
from lab_common.pii.redactor import detect_all

pytestmark = pytest.mark.security

# Printed on the samples (all fictional).
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
    "0000111122223333",
    "João Exemplo",
    "CRM-MG 000000",
    "Rua Imaginária",
]
RUNS = [
    ("pedido_medico_ficticio.png", "pedido_medico_ficticio.json"),
    ("pedido_ambiguo.png", "pedido_ambiguo.json"),
    ("pedido_injecao.png", "pedido_medico_ficticio.json"),
]


def _leaks(text: str) -> list[str]:
    values = [value for value in PII_VALUES if value in text]
    patterns = [
        f"{span.entity.value}:{text[span.start : span.end]!r}"
        for span in detect_all(high_precision_detectors(), text, 0.0)
    ]
    return values + patterns


@pytest.fixture(scope="module")
def runs(e2e_stack: ComposeStack) -> list[AgentRun]:
    """Each sample twice: the JSON report and the table a person reads."""
    return [
        e2e_stack.run_agent(
            f"{SAMPLES}/{image}",
            "--non-interactive",
            *output,
            env={"FAKE_LLM_FIXTURE": f"{FAKE_LLM_FIXTURES}/{fixture}"},
        )
        for image, fixture in RUNS
        for output in (("--output", "json"), ("--output", "table"))
    ]


@pytest.mark.e2e
def test_no_personal_data_in_the_logs_of_any_container(
    e2e_stack: ComposeStack, runs: list[AgentRun]
) -> None:
    services = e2e_stack.logs()
    agent = "".join(run.stderr for run in runs)

    for name in ("scheduler-api", "ocr-mcp", "rag-mcp", "postgres"):
        assert f"{name}-1" in services, f"no logs from {name}"
    assert "run_finished" in agent
    assert _leaks(services) == []
    assert _leaks(agent) == []


@pytest.mark.e2e
def test_no_personal_data_in_the_cli_output(runs: list[AgentRun]) -> None:
    output = "".join(run.stdout for run in runs)

    assert "LAB-10001" in output
    assert _leaks(output) == []


@pytest.mark.e2e
def test_no_personal_data_in_the_database(e2e_stack: ComposeStack, runs: list[AgentRun]) -> None:
    dump = e2e_stack.compose(
        "exec", "-T", "postgres", "pg_dump", "-U", "scheduler", "--data-only", "scheduler"
    ).stdout

    assert runs
    assert "COPY public.appointments" in dump
    assert "pt_" in dump
    assert _leaks(dump) == []


@pytest.mark.e2e
def test_the_image_never_reaches_logs_output_or_database(
    e2e_stack: ComposeStack, runs: list[AgentRun]
) -> None:
    text = e2e_stack.logs() + "".join(run.stdout + run.stderr for run in runs)

    for image, _ in RUNS:
        encoded = base64.b64encode((REPO_ROOT / "samples" / image).read_bytes()).decode()
        assert encoded[:64] not in text
        assert encoded[-64:] not in text
