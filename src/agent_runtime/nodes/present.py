"""`present`: the terminal node of a run that got as far as the gate.

`present` and `abort` both write a `RunResult` to `run.result` (and as their output); the CLI reads
it to print the report and pick the exit code. Nothing here holds personal
data: mentions come from the masked text, the patient is never named and PII
appears only as counts per type.
"""

import unicodedata
from collections.abc import Mapping
from io import StringIO
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode
from pydantic import BaseModel, ConfigDict
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

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
REVIEWED = "reviewed"
REVIEWED_MARK = "✎"
_REVIEW_REASONS = {"evidence_mismatch", "evidence_ambiguous"}
_ATTENTION_REASONS = {"needs_human_review", "evidence_mismatch", "evidence_ambiguous"}
_REASON_LABELS = {
    "not_found": "não encontrado no catálogo",
    "low_confidence": "não encontrado no catálogo",
    "low_margin": "corresponde a mais de um exame do catálogo",
    "below_accept": "correspondência incerta no catálogo",
    "already_accepted": "já incluído por outro item do pedido",
    "rejected_by_reviewer": "descartado por você na revisão",
    "invalid_review_response": "resposta da revisão inválida; item descartado",
    "needs_human_review": "precisa de revisão humana (rode sem --non-interactive)",
    "evidence_mismatch": "precisa de revisão: a interpretação da IA não confere com o pedido",
    "evidence_ambiguous": "precisa de revisão: o texto do pedido serve para mais de um exame",
}
_SUMMARIES: dict[str, tuple[str, str]] = {
    "scheduled": ("✔ Solicitação registrada", "bold green"),
    "replayed": ("✔ Solicitação já registrada antes; nada foi duplicado", "bold green"),
    "resolved": ("Exames identificados no catálogo; este agente não registra solicitação", "bold"),
    "not_scheduled": ("✘ Solicitação não registrada", "bold red"),
    "schedule_failed": ("✘ Solicitação não registrada: o agendamento recusou o pedido", "bold red"),
    "rejected": ("✘ Solicitação não registrada: nenhum exame foi confirmado", "bold red"),
}
_NOT_SCHEDULED_ACTIONS = {
    "patient_identifier_missing": (
        "O pedido não identifica o paciente. Envie um pedido com o CPF ou a carteirinha legível."
    ),
    "no_accepted_exams": "Nenhum exame foi confirmado. Confira o pedido e rode de novo.",
}
_OUTCOME_ACTIONS = {
    "schedule_failed": "Tente de novo; se continuar, rode com --verbose e procure o suporte.",
    "rejected": "Se algum item deveria ser agendado, rode de novo e escolha o exame na revisão.",
}
_COUNT_WORDS = {
    "scheduled": ("agendado", "não agendado"),
    "replayed": ("agendado", "não agendado"),
    "resolved": ("identificado", "não identificado"),
    "not_scheduled": ("confirmado", "não agendado"),
}
_ACCEPTED_TITLES = {
    "scheduled": "Exames agendados",
    "replayed": "Exames agendados",
    "resolved": "Exames identificados",
}
_FAILURES: dict[str, tuple[str, str | None]] = {
    "OCR_LOW_QUALITY": ("✘ Não foi possível ler o pedido", None),
    "INPUT_BLOCKED": (
        "✘ Pedido bloqueado por segurança",
        "Nada foi agendado. Encaminhe o documento original para conferência manual.",
    ),
    "NO_EXAMS_RESOLVED": ("✘ Nenhum exame do pedido foi encontrado no catálogo", None),
}
_PII_LABELS = {
    "PERSON": "nome",
    "ADDRESS": "endereço",
    "BIRTH_DATE": "data de nascimento",
    "PHONE": "telefone",
    "EMAIL": "e-mail",
    "HEALTH_CARD": "carteirinha",
}
_SEVERITY_LABELS = {"low": "baixo", "medium": "médio", "high": "alto"}
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


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _comparable(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c) and not c.isspace())


def as_written(row: ExamRow) -> str:
    """The order's text for an accepted row, only when it differs from the catalog name."""
    if row.name is not None and _comparable(row.evidence) == _comparable(row.name):
        return ""
    return row.evidence


def reason_label(row: ExamRow) -> str:
    """Why a row was not scheduled, in plain Portuguese."""
    label = _REASON_LABELS.get(row.reason or "", "não confirmado")
    if row.status == "review" and row.reason not in _REVIEW_REASONS:
        return f"precisa de revisão humana: {label}"
    return label


def _pii_line(counts: Mapping[str, int]) -> Text:
    total = sum(counts.values())
    if not total:
        return Text("Dados pessoais mascarados: nenhum")
    labels = sorted((_PII_LABELS.get(kind, kind), count) for kind, count in counts.items())
    detail = ", ".join(f"{label} {count}" for label, count in labels)
    return Text(f"Dados pessoais mascarados: {total} ({detail})")


def _summary(result: RunResult) -> Text:
    label, style = _SUMMARIES[result.outcome]
    line = Text(label, style=style)
    if result.appointment and result.appointment.id:
        line.append(f" · id {result.appointment.id}")
    action = _NOT_SCHEDULED_ACTIONS.get(result.reason or "") or _OUTCOME_ACTIONS.get(result.outcome)
    if action:
        line.append(f"\n{action}")
    return line


def _counts(result: RunResult, accepted: int, other: int) -> Text:
    done, missing = _COUNT_WORDS.get(result.outcome, _COUNT_WORDS["not_scheduled"])
    return Text(
        f"{_plural(accepted, 'exame ' + done, 'exames ' + done + 's')} · "
        f"{_plural(other, missing, missing + 's')}"
    )


def _accepted_table(result: RunResult, rows: list[ExamRow]) -> Table:
    title = _ACCEPTED_TITLES.get(result.outcome, "Exames confirmados, não agendados")
    table = Table(
        title=f"{title} ({len(rows)})",
        title_style="bold green",
        title_justify="left",
        border_style="green",
    )
    table.add_column("Exame")
    table.add_column("Código", no_wrap=True)
    table.add_column("Como estava no pedido")
    for row in rows:
        name = Text(row.name or "")
        if row.reason == REVIEWED:
            name.append(f" {REVIEWED_MARK}", style="bold")
        table.add_row(name, Text(row.code or ""), Text(as_written(row), style="dim"))
    if any(row.reason == REVIEWED for row in rows):
        table.caption = f"{REVIEWED_MARK} escolhido por você na revisão"
        table.caption_justify = "left"
    return table


def _other_table(result: RunResult, rows: list[ExamRow]) -> Table:
    missing = _COUNT_WORDS.get(result.outcome, _COUNT_WORDS["not_scheduled"])[1]
    table = Table(
        title=f"{missing.capitalize()}s ({len(rows)})",
        title_style="bold yellow",
        title_justify="left",
        border_style="yellow",
    )
    table.add_column("Como estava no pedido")
    table.add_column("Motivo")
    for row in rows:
        style = "red" if row.reason in _ATTENTION_REASONS else "yellow"
        table.add_row(Text(row.evidence), Text(reason_label(row), style=style))
    return table


def _failure(result: RunResult) -> Group:
    title, action = _FAILURES.get(result.outcome, ("Não foi possível concluir o pedido", None))
    parts = [Text(title, style="bold red"), Text(result.user_message or "")]
    if action:
        parts.append(Text(action))
    return Group(*parts)


def _distinct(rows: list[ExamRow]) -> list[ExamRow]:
    """One row per order text and reason: the model and the lexical pass may read the same line."""
    seen: set[tuple[str, str]] = set()
    kept: list[ExamRow] = []
    for row in rows:
        key = (_comparable(row.evidence), reason_label(row))
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def _injection_warning(severity: str) -> Text:
    label = _SEVERITY_LABELS.get(severity, severity)
    return Text(
        f"Atenção: o pedido tem trechos que parecem instruções (risco {label}).", style="bold red"
    )


def result_view(result: RunResult) -> Group:
    """What the operator reads: a summary, the scheduled exams and the ones left out."""
    warning = [_injection_warning(result.injection_severity)] * (
        result.injection_severity != "none"
    )
    if result.user_message:
        return Group(_failure(result), _pii_line(result.pii_masked), *warning)
    accepted = [row for row in result.exams if row.status == "accepted"]
    other = _distinct([row for row in result.exams if row.status != "accepted"])
    parts: list[RenderableType] = [
        _summary(result),
        _counts(result, len(accepted), len(other)),
        _pii_line(result.pii_masked),
        *warning,
    ]
    if accepted:
        parts += [Text(), _accepted_table(result, accepted)]
    if other:
        parts += [Text(), _other_table(result, other)]
    return Group(*parts)


def details_view(result: RunResult) -> Group:
    """Every exam with score, status and technical reason (`--verbose`)."""
    parts: list[RenderableType] = []
    if result.user_message:
        parts.append(Text(f"Resultado: {result.outcome}", style="bold"))
        parts.append(Text(result.user_message))
    else:
        table = Table(title="Exames do pedido", show_lines=False)
        for column in ("No pedido", "Exame", "Código", "Nome no catálogo", "Score", "Status"):
            table.add_column(column)
        table.add_column("Motivo")
        for row in result.exams:
            table.add_row(
                Text(row.evidence),
                Text(row.mention),
                row.code or "—",
                Text(row.name or "—"),
                f"{row.score:.4f}" if row.score is not None else "—",
                _STATUS_LABELS[row.status],
                row.reason or "",
            )
        parts.append(table)
        label = _OUTCOME_LABELS.get(result.outcome, result.outcome)
        line = f"Agendamento: {label}"
        if result.appointment and result.appointment.id:
            line += f" — id {result.appointment.id}"
        if result.reason:
            line += f" ({result.reason})"
        parts.append(Text(line))
    masked = ", ".join(f"{k} {v}" for k, v in sorted(result.pii_masked.items())) or "nenhum"
    parts.append(Text(f"Dados pessoais mascarados: {masked}"))
    if result.injection_severity != "none":
        parts.append(
            Text(
                f"Atenção: conteúdo suspeito no documento (severidade {result.injection_severity})."
            )
        )
    return Group(*parts)


def to_text(renderable: RenderableType, width: int = 120, *, color: bool = False) -> str:
    """`renderable` as terminal text; with `color`, as ANSI sequences."""
    console = Console(
        file=StringIO(),
        width=width,
        color_system="standard" if color else None,
        force_terminal=color,
        no_color=not color,
        highlight=False,
    )
    console.print(renderable)
    file = console.file
    text = file.getvalue() if isinstance(file, StringIO) else ""
    return text if color else "\n".join(line.rstrip() for line in text.split("\n"))


def render_table(result: RunResult, width: int = 120, *, verbose: bool = False) -> str:
    """Plain-text report for the operator; `verbose` adds the technical columns."""
    return to_text(details_view(result) if verbose else result_view(result), width)


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
