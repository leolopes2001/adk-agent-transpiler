"""Prompt injection against the whole flow (real OCR, RAG and API; FakeLlm).

The worst a successful injection may do is make the model omit an exam: it
cannot reach the model past the guard, cannot add exams or codes and cannot
trigger a scheduling request by itself.
"""

import json
from pathlib import Path

import pytest

from runtime_stack import REPO_ROOT, Stack, run_agent

pytestmark = pytest.mark.security

SAMPLE = "samples/pedido_medico_ficticio.png"
EXPECTED_CODES = [
    exam["code"]
    for exam in json.loads(
        (REPO_ROOT / "samples" / "expected" / "pedido_medico_ficticio.json").read_text("utf-8")
    )["exams"]
]


def _accepted(report: dict[str, object]) -> list[str]:
    exams = report["exams"]
    assert isinstance(exams, list)
    return [e["code"] for e in exams if e["status"] == "accepted"]


async def test_injection_sample_never_reaches_the_model_or_the_api(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = stack.count("idempotency_keys")
    appointments = stack.count("appointments")

    run = await run_agent(stack, monkeypatch, "samples/pedido_injecao.png")

    assert run.exit_code == 3
    assert run.llm_requests == []
    assert (stack.count("idempotency_keys"), stack.count("appointments")) == (keys, appointments)
    assert '"injection_detected"' in run.logs
    assert "IGNORE" not in run.logs


async def test_model_that_obeys_an_injection_cannot_add_exams(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model "complied": an exam the order does not have, a placeholder as a name.
    run = await run_agent(stack, monkeypatch, SAMPLE, fixture="hallucinated.json")

    assert run.exit_code == 0
    assert _accepted(run.report) == EXPECTED_CODES
    [rejected] = [
        json.loads(line) for line in run.logs.splitlines() if '"grounding_rejected"' in line
    ]
    assert (rejected["ungrounded"], rejected["invalid_name"]) == (1, 1)


async def test_codes_written_by_the_model_are_not_codes(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = tmp_path / "codes.json"
    fixture.write_text(
        json.dumps({"exams": [{"name": "LAB-10099", "evidence": "1. Hemograma completo"}]})
    )

    run = await run_agent(
        stack, monkeypatch, "samples/pedido_medico_ficticio.png", fixture=str(fixture)
    )

    assert "LAB-10099" not in _accepted(run.report)
    assert _accepted(run.report) == EXPECTED_CODES
