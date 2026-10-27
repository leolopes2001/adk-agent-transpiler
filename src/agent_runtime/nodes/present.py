"""`present`: the terminal node of a run that got as far as the gate.

`present` and `abort` both write a `RunResult` to `run.result` (and as their output); the CLI reads
it to print the report and pick the exit code. Nothing here holds personal
data: mentions come from the masked text, the patient is never named and PII
appears only as counts per type.
"""

from io import StringIO
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from agent_runtime.nodes._common import Params, function_node, parse_input
from agent_runtime.schemas import GateDecision, GateItem, RunFailure, ScheduleOutcome

RESULT_STATE_KEY = "run.result"

Outcome = Literal[
    "scheduled",
    "replayed",
    "not_scheduled",
    "schedule_failed",
    "resolved",
    "rejected",
    "OCR_LOW_QUALITY",
    "INPUT_BLOCKED",
    "NO_EXAMS_RESOLVED",
]
_SCHEDULE_OUTCOMES: dict[str, Outcome] = {
    "scheduled": "scheduled",
    "replayed": "replayed",
    "not_scheduled": "not_scheduled",
    "failed": "schedule_failed",
}
_STATUS_LABELS = {"accepted": "aceito", "review": "revisão", "rejected": "rejeitado"}
_OUTCOME_LABELS = {
    "scheduled": "solicitação registrada",
    "replayed": "solicitação já existente (replay idempotente)",
    "not_scheduled": "não agendado",
    "schedule_failed": "agendamento recusado",
    "resolved": "exames identificados (sem agendamento)",
    "rejected": "nenhum exame confirmado",
}


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExamRow(_Result):
    mention: str
    evidence: str
    source: Literal["llm", "lexical", "both"]
    status: Literal["accepted", "review", "rejected"]
    code: str | None
    name: str | None
    score: float | None
    reason: str | None


class Appointment(_Result):
    id: str | None
    status: Literal["scheduled", "replayed", "not_scheduled", "failed"]


class RunResult(_Result):
    outcome: Outcome
    format: Literal["table", "json"]
    exams: list[ExamRow]
    appointment: Appointment | None
    reason: str | None
    user_message: str | None
    pii_masked: dict[str, int]
    injection_severity: str


class PresentParams(Params):
    format: Literal["table", "json"] = "table"


def exam_row(item: GateItem) -> ExamRow:
    shown = item.selected
    scored = shown or (item.candidates[0] if item.candidates else None)
    return ExamRow(
        mention=item.mention.text,
        evidence=item.mention.evidence,
        source=item.mention.source,
        status=item.status,
        code=shown.code if shown else None,
        name=shown.name if shown else None,
        score=scored.score if scored else None,
        reason=item.reason,
    )


def _context(ctx: Context) -> dict[str, Any]:
    counts = ctx.state.get("run.pii_counts") or {}
    return {
        "pii_masked": {str(k): int(v) for k, v in dict(counts).items()},
        "injection_severity": str(ctx.state.get("run.injection_severity") or "none"),
    }


OutputFormat = Literal["table", "json"]


def present_result(
    node_input: object, output_format: OutputFormat, context: dict[str, Any]
) -> RunResult:
    """Result for a schedule outcome, or for a gate decision when nothing was scheduled."""
    if isinstance(node_input, ScheduleOutcome) or (
        isinstance(node_input, dict) and "decision" in node_input
    ):
        outcome = parse_input(ScheduleOutcome, node_input)
        decision = outcome.decision
        appointment: Appointment | None = Appointment(
            id=outcome.appointment_id, status=outcome.status
        )
        result_outcome = _SCHEDULE_OUTCOMES[outcome.status]
        reason = outcome.reason
    else:
        decision = parse_input(GateDecision, node_input)
        appointment = None
        accepted = any(item.status == "accepted" for item in decision.items)
        result_outcome = "resolved" if accepted else "rejected"
        reason = None
    return RunResult(
        outcome=result_outcome,
        format=output_format,
        exams=[exam_row(item) for item in decision.items],
        appointment=appointment,
        reason=reason,
        user_message=None,
        **context,
    )


def run_context(ctx: Context) -> dict[str, Any]:
    """PII counts and injection severity the OCR and guard nodes left in the state."""
    return _context(ctx)


def failure_result(failure: RunFailure, context: dict[str, Any]) -> RunResult:
    return RunResult(
        outcome=failure.code,
        format="table",
        exams=[],
        appointment=None,
        reason=None,
        user_message=failure.user_message,
        **context,
    )


def render_json(result: RunResult) -> str:
    return result.model_dump_json(indent=2)


def render_table(result: RunResult, width: int = 120) -> str:
    """Plain-text table of the exams plus a footer with the appointment and PII counts."""
    console = Console(file=StringIO(), width=width, color_system=None, record=True)
    if result.user_message:
        console.print(f"[bold]Resultado:[/bold] {result.outcome}")
        console.print(result.user_message, markup=False)
    else:
        table = Table(title="Exames do pedido", show_lines=False)
        for column in ("No pedido", "Exame", "Código", "Nome no catálogo", "Score", "Status"):
            table.add_column(column)
        table.add_column("Motivo")
        for row in result.exams:
            table.add_row(
                row.evidence,
                row.mention,
                row.code or "—",
                row.name or "—",
                f"{row.score:.4f}" if row.score is not None else "—",
                _STATUS_LABELS[row.status],
                row.reason or "",
            )
        console.print(table)
        label = _OUTCOME_LABELS.get(result.outcome, result.outcome)
        line = f"Agendamento: {label}"
        if result.appointment and result.appointment.id:
            line += f" — id {result.appointment.id}"
        if result.reason:
            line += f" ({result.reason})"
        console.print(line, markup=False)
    masked = ", ".join(f"{k} {v}" for k, v in sorted(result.pii_masked.items())) or "nenhum"
    console.print(f"Dados pessoais mascarados: {masked}", markup=False)
    if result.injection_severity != "none":
        console.print(
            f"Atenção: conteúdo suspeito no documento (severidade {result.injection_severity}).",
            markup=False,
        )
    return console.export_text()


def render(result: RunResult, output_format: str | None = None) -> str:
    chosen = output_format or result.format
    return render_json(result) if chosen == "json" else render_table(result)


def present(*, name: str, format: Literal["table", "json"] = "table") -> FunctionNode:
    """Renders the final result."""
    params = PresentParams(format=format)

    async def run(ctx: Context, node_input: Any) -> Event:
        result = present_result(node_input, params.format, run_context(ctx))
        ctx.state[RESULT_STATE_KEY] = result.model_dump(mode="json")
        return Event(output=result)

    return function_node(name, run)
