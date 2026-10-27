import json
import re
from typing import Any

import pytest
from runtime_fakes import candidate, run_node

from agent_runtime.nodes.abort import abort
from agent_runtime.nodes.present import (
    REVIEWED_MARK,
    ExamRow,
    RunResult,
    as_written,
    present,
    reason_label,
    render,
    render_json,
    render_table,
    result_view,
    to_text,
)
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


async def test_table_summarises_and_splits_scheduled_from_not_scheduled() -> None:
    state = {**STATE, "run.injection_severity": "medium"}
    run = await run_node(present(name="present"), _outcome("scheduled", APPOINTMENT_ID), state)

    table = render_table(RunResult.model_validate(run.output))

    assert f"✔ Solicitação registrada · id {APPOINTMENT_ID}" in table
    assert "1 exame agendado · 1 não agendado" in table
    assert "Dados pessoais mascarados: 3 (CPF 1, nome 2)" in table
    assert "(risco médio)" in table
    agendados, nao_agendados = table.split("Exames agendados (1)")[1].split("Não agendados (1)")
    assert "TSH" in agendados
    assert "LAB-10043" in agendados
    assert "3. T4" in nao_agendados
    assert "precisa de revisão humana (rode sem --non-interactive)" in nao_agendados
    for technical in ("1.0000", "0.8000", "needs_human_review", "aceito", "rejeitado", "Score"):
        assert technical not in table


def _row(**fields: Any) -> ExamRow:
    base = {
        "mention": "x",
        "evidence": "x",
        "source": "both",
        "status": "rejected",
        "code": None,
        "name": None,
        "score": 0.5,
        "reason": None,
    }
    return ExamRow.model_validate(base | fields)


@pytest.mark.parametrize(
    ("evidence", "name", "shown"),
    [
        ("HEMOGRAMA  completo", "Hemograma completo", ""),
        ("Proteina C reativa", "Proteína C reativa", ""),
        ("PCR", "Proteína C reativa", "PCR"),
        ("T4", "T4 livre", "T4"),
    ],
)
def test_as_written_only_when_it_differs_from_the_catalog_name(
    evidence: str, name: str, shown: str
) -> None:
    row = _row(evidence=evidence, name=name, code="LAB-1", status="accepted")

    assert as_written(row) == shown


REASONS = [
    "not_found",
    "low_confidence",
    "already_accepted",
    "rejected_by_reviewer",
    "invalid_review_response",
    "needs_human_review",
    "evidence_mismatch",
    "evidence_ambiguous",
    "low_margin",
    "below_accept",
    "unknown_reason",
]


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("status", ["rejected", "review"])
def test_every_reason_reads_as_portuguese(reason: str, status: str) -> None:
    label = reason_label(_row(reason=reason, status=status))

    assert reason not in label
    assert "_" not in label


def test_reason_labels() -> None:
    assert reason_label(_row(reason="low_confidence")) == "não encontrado no catálogo"
    assert reason_label(_row(reason="already_accepted")) == "já incluído por outro item do pedido"
    assert reason_label(_row(reason="rejected_by_reviewer")) == "descartado por você na revisão"
    assert reason_label(_row(reason="low_margin", status="review")) == (
        "precisa de revisão humana: corresponde a mais de um exame do catálogo"
    )


def _result(outcome: str, *rows: ExamRow, **fields: Any) -> RunResult:
    base = {
        "outcome": outcome,
        "format": "table",
        "exams": list(rows),
        "appointment": None,
        "reason": None,
        "user_message": None,
        "pii_masked": {},
        "injection_severity": "none",
    }
    return RunResult.model_validate(base | fields)


def test_reviewed_rows_are_marked_with_a_legend() -> None:
    reviewed = _row(
        evidence="T4", name="T4 total", code="LAB-10045", status="accepted", reason="reviewed"
    )
    plain = _row(evidence="TSH", name="TSH", code="LAB-10043", status="accepted")

    marked = render_table(_result("scheduled", reviewed, plain))
    unmarked = render_table(_result("scheduled", plain))

    assert f"T4 total {REVIEWED_MARK}" in marked
    assert f"{REVIEWED_MARK} escolhido por você na revisão" in marked
    assert REVIEWED_MARK not in unmarked


@pytest.mark.parametrize(
    ("outcome", "fields", "expected"),
    [
        ("replayed", {}, "Solicitação já registrada antes; nada foi duplicado"),
        ("resolved", {}, "1 exame identificado · 0 não identificados"),
        ("rejected", {}, "rode de novo e escolha o exame na revisão"),
        ("schedule_failed", {"reason": "Unknown exam code"}, "o agendamento recusou o pedido"),
        (
            "not_scheduled",
            {"reason": "patient_identifier_missing"},
            "O pedido não identifica o paciente.",
        ),
    ],
)
def test_summary_per_outcome(outcome: str, fields: dict[str, Any], expected: str) -> None:
    row = _row(evidence="TSH", name="TSH", code="LAB-10043", status="accepted")

    table = render_table(_result(outcome, row, **fields))

    assert expected in table
    assert "Unknown exam code" not in table
    assert "patient_identifier_missing" not in table


@pytest.mark.parametrize(
    ("code", "title"),
    [
        ("OCR_LOW_QUALITY", "Não foi possível ler o pedido"),
        ("INPUT_BLOCKED", "Pedido bloqueado por segurança"),
        ("NO_EXAMS_RESOLVED", "Nenhum exame do pedido foi encontrado no catálogo"),
    ],
)
async def test_table_for_a_failure_explains_it(code: str, title: str) -> None:
    failure = RunFailure(code=code, user_message="Envie uma imagem mais nítida.")  # type: ignore[arg-type]
    run = await run_node(abort(name="abort"), failure)

    table = render_table(RunResult.model_validate(run.output))

    assert title in table
    assert "Envie uma imagem mais nítida." in table
    assert "Dados pessoais mascarados: nenhum" in table
    assert code not in table


async def test_verbose_keeps_the_technical_table() -> None:
    run = await run_node(present(name="present"), _outcome("scheduled", APPOINTMENT_ID), STATE)

    table = render_table(RunResult.model_validate(run.output), verbose=True)

    for expected in ("3. TSH", "LAB-10043", "1.0000", "aceito", "rejeitado", "0.8000"):
        assert expected in table
    assert "needs_human_review" in table
    assert f"id {APPOINTMENT_ID}" in table
    assert "CPF 1, PERSON 2" in table


def test_colour_is_optional_and_the_text_is_the_same() -> None:
    row = _row(evidence="TSH", name="TSH", code="LAB-10043", status="accepted")
    view = result_view(_result("scheduled", row))

    coloured = to_text(view, color=True)
    plain = to_text(view)

    assert "\x1b[" in coloured
    assert "\x1b[" not in plain
    assert re.sub(r"\x1b\[[0-9;]*m", "", coloured).split() == plain.split()


def test_the_same_order_text_with_the_same_reason_is_listed_once() -> None:
    llm = _row(mention="T4 total", evidence="T4", reason="needs_human_review")
    lexical = _row(mention="T4", evidence="T4", reason="needs_human_review")
    other = _row(mention="T4", evidence="T4", reason="already_accepted")

    table = render_table(_result("scheduled", llm, lexical, other))

    assert "0 exames agendados · 2 não agendados" in table
    assert "Não agendados (2)" in table
    assert table.count("precisa de revisão humana") == 1


def test_document_text_is_not_markup() -> None:
    row = _row(evidence="[red]X[/red]", reason="low_confidence")

    assert "[red]X[/red]" in render_table(_result("scheduled", row))


async def test_json_round_trips_and_render_honours_the_format() -> None:
    run = await run_node(present(name="present"), _outcome("replayed", APPOINTMENT_ID), STATE)
    result = RunResult.model_validate(run.output)

    assert RunResult.model_validate(json.loads(render_json(result))) == result
    assert render(result) == render_table(result)
    assert render(result, "json") == render_json(result)


def test_format_is_validated() -> None:
    with pytest.raises(ValueError, match="format"):
        present(name="present", format="html")  # type: ignore[arg-type]
