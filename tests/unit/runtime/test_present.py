import json

import pytest
from runtime_fakes import candidate, run_node

from agent_runtime.nodes.abort import abort
from agent_runtime.nodes.present import RunResult, present, render, render_json, render_table
from agent_runtime.schemas import (
    Candidate,
    ExamMention,
    GateDecision,
    GateItem,
    RunFailure,
    ScheduleOutcome,
)

pytestmark = pytest.mark.unit

APPOINTMENT_ID = "apt_01JABCDEFGHJKMNPQRSTVWXYZ0"
STATE = {"run.pii_counts": {"PERSON": 2, "CPF": 1}, "run.injection_severity": "none"}
TSH = Candidate.model_validate(candidate("LAB-10043", "TSH", 1.0))
T4 = Candidate.model_validate(candidate("LAB-10044", "T4 livre", 0.8))


def _item(text: str, status: str, selected: Candidate | None, *others: Candidate) -> GateItem:
    candidates = [c for c in (selected, *others) if c is not None]
    return GateItem.model_validate(
        {
            "mention": ExamMention(text=text, source="both", evidence=f"3. {text}"),
            "status": status,
            "selected": selected,
            "candidates": candidates,
            "reason": None if status == "accepted" else "needs_human_review",
        }
    )


DECISION = GateDecision(items=[_item("TSH", "accepted", TSH), _item("T4", "rejected", None, T4)])


def _outcome(status: str, appointment_id: str | None, reason: str | None = None) -> ScheduleOutcome:
    return ScheduleOutcome(
        decision=DECISION,
        status=status,
        appointment_id=appointment_id,
        reason=reason,  # type: ignore[arg-type]
    )


async def test_scheduled_result_is_written_to_state() -> None:
    run = await run_node(present(name="present"), _outcome("scheduled", APPOINTMENT_ID), STATE)

    result = RunResult.model_validate(run.state["run.result"])
    assert result == RunResult.model_validate(run.output)
    assert result.outcome == "scheduled"
    assert result.appointment is not None
    assert (result.appointment.id, result.appointment.status) == (APPOINTMENT_ID, "scheduled")
    assert [(r.mention, r.code, r.status, r.score) for r in result.exams] == [
        ("TSH", "LAB-10043", "accepted", 1.0),
        ("T4", None, "rejected", 0.8),
    ]
    assert result.pii_masked == {"PERSON": 2, "CPF": 1}


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("replayed", "replayed"), ("not_scheduled", "not_scheduled"), ("failed", "schedule_failed")],
)
async def test_schedule_statuses(status: str, outcome: str) -> None:
    run = await run_node(present(name="present"), _outcome(status, None, "x"), STATE)

    assert RunResult.model_validate(run.output).outcome == outcome


async def test_gate_decision_without_schedule() -> None:
    resolved = await run_node(present(name="present", format="json"), DECISION, STATE)
    rejected = await run_node(present(name="present"), GateDecision(items=[DECISION.items[1]]))

    result = RunResult.model_validate(resolved.output)
    assert (result.outcome, result.format, result.appointment) == ("resolved", "json", None)
    assert RunResult.model_validate(rejected.output).outcome == "rejected"


async def test_abort_records_the_failure() -> None:
    failure = RunFailure(code="INPUT_BLOCKED", user_message="Revisão manual necessária.")

    run = await run_node(abort(name="abort"), failure, {"run.injection_severity": "high"})

    result = RunResult.model_validate(run.state["run.result"])
    assert (result.outcome, result.user_message, result.exams) == (
        "INPUT_BLOCKED",
        "Revisão manual necessária.",
        [],
    )
    assert result.injection_severity == "high"
    assert result.pii_masked == {}


async def test_table_shows_exams_appointment_pii_and_injection_warning() -> None:
    state = {**STATE, "run.injection_severity": "medium"}
    run = await run_node(present(name="present"), _outcome("scheduled", APPOINTMENT_ID), state)

    table = render_table(RunResult.model_validate(run.output))

    for expected in ("3. TSH", "LAB-10043", "1.0000", "aceito", "rejeitado", "0.8000"):
        assert expected in table
    assert f"id {APPOINTMENT_ID}" in table
    assert "CPF 1, PERSON 2" in table
    assert "severidade medium" in table


async def test_table_for_a_failure_shows_the_message() -> None:
    failure = RunFailure(code="OCR_LOW_QUALITY", user_message="Envie uma imagem mais nítida.")
    run = await run_node(abort(name="abort"), failure)

    table = render_table(RunResult.model_validate(run.output))

    assert "OCR_LOW_QUALITY" in table
    assert "Envie uma imagem mais nítida." in table
    assert "Dados pessoais mascarados: nenhum" in table


async def test_json_round_trips_and_render_honours_the_format() -> None:
    run = await run_node(present(name="present"), _outcome("replayed", APPOINTMENT_ID), STATE)
    result = RunResult.model_validate(run.output)

    assert RunResult.model_validate(json.loads(render_json(result))) == result
    assert render(result) == render_table(result)
    assert render(result, "json") == render_json(result)


def test_format_is_validated() -> None:
    with pytest.raises(ValueError, match="format"):
        present(name="present", format="html")  # type: ignore[arg-type]
