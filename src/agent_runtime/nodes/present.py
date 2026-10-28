import unicodedata
from collections.abc import Mapping
from io import StringIO
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.workflow import FunctionNode
from pydantic import BaseModel, ConfigDict
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from agent_runtime.nodes._common import Params, function_node, parse_input
from agent_runtime.schemas import GateDecision, GateItem, RunFailure, ScheduleOutcome
from agent_runtime.theme import (
    FAIL,
    ICON_FAIL,
    ICON_OK,
    ICON_REVIEWED,
    LABEL,
    OK,
    THEME,
    WARN,
)

RESULT_STATE_KEY = "run.result"
REPORT_WIDTH = 120
UNFINISHED_TITLE = f"{ICON_FAIL} Não foi possível concluir o pedido"
REVIEWED = "reviewed"
REVIEWED_MARK = ICON_REVIEWED
SUMMARY_LABEL_WIDTH = 18

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
OutputFormat = Literal["table", "json"]
Pair = tuple[str | Text, Text | str]

_SCHEDULE_OUTCOMES: dict[str, Outcome] = {
    "scheduled": "scheduled",
    "replayed": "replayed",
    "not_scheduled": "not_scheduled",
    "failed": "schedule_failed",
}
_REVIEW_REASONS = {"evidence_mismatch", "evidence_ambiguous"}
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
_NOT_REGISTERED = (f"{ICON_FAIL} Solicitação não registrada", False)
_SUMMARY_BY_OUTCOME: dict[str, tuple[str, bool]] = {
    "scheduled": (f"{ICON_OK} Solicitação registrada", True),
    "replayed": (f"{ICON_OK} Solicitação já registrada antes (nada foi duplicado)", True),
    "resolved": (f"{ICON_OK} Exames identificados no catálogo", True),
    "not_scheduled": _NOT_REGISTERED,
    "schedule_failed": _NOT_REGISTERED,
    "rejected": _NOT_REGISTERED,
}
_WHY_NOT_SCHEDULED = {
    "patient_identifier_missing": (
        "o pedido não identifica o paciente",
        "Envie um pedido com o CPF ou a carteirinha legível.",
    ),
    "no_accepted_exams": (
        "nenhum exame foi confirmado",
        "Confira o pedido e rode de novo.",
    ),
    "schedule_failed": (
        "o sistema de agendamento recusou o pedido",
        "Tente de novo; se continuar, rode com --verbose e procure o suporte.",
    ),
    "rejected": (
        "nenhum exame foi confirmado na revisão",
        "Se algum item deveria ser agendado, rode de novo e escolha o exame na revisão.",
    ),
}
_SCHEDULED_COUNT_LABELS = ("Agendados", "Não agendados")
_COUNT_LABELS = {
    "scheduled": _SCHEDULED_COUNT_LABELS,
    "replayed": _SCHEDULED_COUNT_LABELS,
    "resolved": ("Identificados", "Não identificados"),
}
_CONFIRMED_LABELS = ("Confirmados", "Não agendados")
_ACCEPTED_TITLES = {
    "scheduled": "Exames agendados",
    "replayed": "Exames agendados",
    "resolved": "Exames identificados",
}
_FAILURE_SUMMARIES: dict[str, tuple[str, str, str]] = {
    "OCR_LOW_QUALITY": (
        f"{ICON_FAIL} Não foi possível ler o pedido",
        "a imagem não tem qualidade suficiente para a leitura",
        "Envie uma imagem mais nítida, sem cortes e bem iluminada.",
    ),
    "INPUT_BLOCKED": (
        f"{ICON_FAIL} Pedido bloqueado por segurança",
        "o documento tem trechos que parecem instruções para a IA",
        "Nada foi agendado. Encaminhe o documento original para conferência manual.",
    ),
    "NO_EXAMS_RESOLVED": (
        f"{ICON_FAIL} Nenhum exame do pedido foi encontrado no catálogo",
        "nenhum item do pedido corresponde, com confiança, a um exame do catálogo",
        "Confira o pedido ou procure o atendimento.",
    ),
}
_PII_NAMES = {
    "PERSON": ("nome", "nomes"),
    "ADDRESS": ("endereço", "endereços"),
    "BIRTH_DATE": ("data de nascimento", "datas de nascimento"),
    "PHONE": ("telefone", "telefones"),
    "EMAIL": ("e-mail", "e-mails"),
    "HEALTH_CARD": ("carteirinha", "carteirinhas"),
    "CPF": ("CPF", "CPFs"),
    "RG": ("RG", "RGs"),
    "CRM": ("CRM", "CRMs"),
    "CEP": ("CEP", "CEPs"),
    "CNS": ("CNS", "CNS"),
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


# Masked mentions, catalog data and PII counts per type only: the patient is never named.
class RunResult(_Result):
    outcome: Outcome
    format: OutputFormat
    exams: list[ExamRow]
    appointment: Appointment | None
    reason: str | None
    user_message: str | None
    pii_masked: dict[str, int]
    injection_severity: str


class PresentParams(Params):
    format: OutputFormat = "table"


def present(*, name: str, format: OutputFormat = "table") -> FunctionNode:
    params = PresentParams(format=format)

    async def run(ctx: Context, node_input: Any) -> Event:
        result = present_result(node_input, params.format, run_context(ctx))
        ctx.state[RESULT_STATE_KEY] = result.model_dump(mode="json")
        return Event(output=result)

    return function_node(name, run)


def present_result(
    node_input: object, output_format: OutputFormat, context: dict[str, Any]
) -> RunResult:
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


def run_context(ctx: Context) -> dict[str, Any]:
    counts = ctx.state.get("run.pii_counts") or {}
    return {
        "pii_masked": {str(k): int(v) for k, v in dict(counts).items()},
        "injection_severity": str(ctx.state.get("run.injection_severity") or "none"),
    }


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


def render(result: RunResult, output_format: str | None = None) -> str:
    chosen = output_format or result.format
    return render_json(result) if chosen == "json" else render_table(result)


def render_json(result: RunResult) -> str:
    return result.model_dump_json(indent=2)


def render_table(result: RunResult, width: int = REPORT_WIDTH, *, verbose: bool = False) -> str:
    return to_text(details_view(result) if verbose else result_view(result), width)


def to_text(renderable: RenderableType, width: int = REPORT_WIDTH, *, color: bool = False) -> str:
    console = Console(
        file=StringIO(),
        width=width,
        color_system="standard" if color else None,
        force_terminal=color,
        no_color=not color,
        highlight=False,
        theme=THEME,
    )
    console.print(renderable)
    file = console.file
    text = file.getvalue() if isinstance(file, StringIO) else ""
    return text if color else "\n".join(line.rstrip() for line in text.split("\n"))


def result_view(result: RunResult) -> Group:
    if result.user_message:
        return _failure_summary(result)

    accepted = [row for row in result.exams if row.status == "accepted"]
    other = _one_per_text_and_reason([row for row in result.exams if row.status != "accepted"])
    parts: list[RenderableType] = [_outcome_summary(result, len(accepted), len(other))]
    if accepted:
        parts += [Text(), _accepted_table(result, accepted)]
    if other:
        parts += [Text(), _not_accepted_table(result, other)]

    return Group(*parts)


def details_view(result: RunResult) -> Group:
    parts: list[RenderableType] = []
    if result.user_message:
        parts.append(Text(f"Resultado: {result.outcome}", style=LABEL))
        parts.append(Text(result.user_message))
    else:
        parts.append(_exams_table(result))
        parts.append(Text(_appointment_line(result)))

    masked = ", ".join(f"{k} {v}" for k, v in sorted(result.pii_masked.items())) or "nenhum"
    parts.append(Text(f"Dados pessoais mascarados: {masked}"))
    if result.injection_severity != "none":
        parts.append(
            Text(
                f"Atenção: conteúdo suspeito no documento (severidade {result.injection_severity})."
            )
        )

    return Group(*parts)


def summary(title: str, style: str, pairs: list[Pair]) -> Group:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=SUMMARY_LABEL_WIDTH, no_wrap=True)
    grid.add_column()
    for label, value in pairs:
        grid.add_row(
            label if isinstance(label, Text) else Text(label, style=LABEL),
            value if isinstance(value, Text) else Text(value),
        )

    return Group(Text(), Text(title, style=style), Padding(grid, (0, 0, 0, 2)))


def matches_catalog(row: ExamRow) -> bool:
    return row.name is not None and _comparable_form(row.evidence) == _comparable_form(row.name)


def as_written(row: ExamRow) -> Text:
    return Text(row.evidence, style="" if matches_catalog(row) else WARN)


def reason_label(row: ExamRow) -> str:
    label = _REASON_LABELS.get(row.reason or "", "não confirmado")
    if row.status == "review" and row.reason not in _REVIEW_REASONS:
        return f"precisa de revisão humana: {label}"

    return label


def masked_summary(counts: Mapping[str, int]) -> str:
    total = sum(counts.values())
    if not total:
        return "nenhum"

    named = [(count, _PII_NAMES.get(kind, (kind, kind))) for kind, count in counts.items() if count]
    named.sort(key=lambda pair: (-pair[0], pair[1][0].casefold()))
    detail = ", ".join(_plural(count, *names) for count, names in named)
    return f"{total} ({detail})"


def _outcome_summary(result: RunResult, accepted: int, other: int) -> Group:
    title, registered = _SUMMARY_BY_OUTCOME[result.outcome]
    pairs: list[Pair] = []
    if result.appointment and result.appointment.id:
        pairs.append(("ID", result.appointment.id))
    done, missing = _COUNT_LABELS.get(result.outcome, _CONFIRMED_LABELS)
    pairs += [(done, _exam_count(accepted)), (missing, _exam_count(other))]
    pairs.append(("Dados mascarados", masked_summary(result.pii_masked)))
    if result.outcome == "resolved":
        pairs.append(("Observação", "este agente não registra solicitação"))
    pairs += _attention_pairs(result.injection_severity)

    why_and_what = _WHY_NOT_SCHEDULED.get(result.reason or "") or _WHY_NOT_SCHEDULED.get(
        result.outcome
    )
    if not registered and why_and_what:
        why, what = why_and_what
        pairs.insert(0, ("Motivo", why))
        pairs.append(("O que fazer", what))

    return summary(title, OK if registered else FAIL, pairs)


def _failure_summary(result: RunResult) -> Group:
    message = result.user_message or ""
    title, why, what = _FAILURE_SUMMARIES.get(result.outcome, (UNFINISHED_TITLE, message, ""))
    pairs: list[Pair] = [("Motivo", why)]
    pairs.append(("Dados mascarados", masked_summary(result.pii_masked)))
    pairs += _attention_pairs(result.injection_severity)
    if what:
        pairs.append(("O que fazer", what))

    return summary(title, FAIL, pairs)


def _attention_pairs(severity: str) -> list[Pair]:
    if severity == "none":
        return []

    label = _SEVERITY_LABELS.get(severity, severity)
    text = f"o pedido tem trechos que parecem instruções (risco {label})"
    return [(Text("Atenção", style=f"{LABEL} {WARN}"), text)]


def _accepted_table(result: RunResult, rows: list[ExamRow]) -> Table:
    title = _ACCEPTED_TITLES.get(result.outcome, "Exames confirmados, não agendados")
    table = Table(
        title=f"{title} ({len(rows)})",
        title_style=LABEL,
        title_justify="left",
        border_style="none",
    )
    table.add_column("Exame")
    table.add_column("Código", no_wrap=True)
    table.add_column("Como estava no pedido")
    for row in rows:
        name = Text(row.name or "")
        if row.reason == REVIEWED:
            name.append(f" {REVIEWED_MARK}", style=WARN)
        table.add_row(name, Text(row.code or ""), as_written(row))

    if any(row.reason == REVIEWED for row in rows):
        table.caption = Text.assemble((REVIEWED_MARK, WARN), " escolhido por você na revisão")
        table.caption_justify = "left"

    return table


def _not_accepted_table(result: RunResult, rows: list[ExamRow]) -> Table:
    missing = _COUNT_LABELS.get(result.outcome, _CONFIRMED_LABELS)[1]
    table = Table(
        title=f"{missing} ({len(rows)})",
        title_style=LABEL,
        title_justify="left",
        border_style="none",
    )
    table.add_column("Como estava no pedido")
    table.add_column("Motivo")
    for row in rows:
        table.add_row(Text(row.evidence), Text(reason_label(row)))

    return table


def _exams_table(result: RunResult) -> Table:
    table = Table(title="Exames do pedido", show_lines=False)
    for column in ("No pedido", "Exame", "Código", "Nome no catálogo", "Score", "Status", "Motivo"):
        table.add_column(column)
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

    return table


def _appointment_line(result: RunResult) -> str:
    label = _OUTCOME_LABELS.get(result.outcome, result.outcome)
    text = f"Agendamento: {label}"
    if result.appointment and result.appointment.id:
        text += f" — id {result.appointment.id}"
    if result.reason:
        text += f" ({result.reason})"

    return text


# The model and the lexical pass may both read the same line of the order.
def _one_per_text_and_reason(rows: list[ExamRow]) -> list[ExamRow]:
    seen: set[tuple[str, str]] = set()
    kept: list[ExamRow] = []
    for row in rows:
        key = (_comparable_form(row.evidence), reason_label(row))
        if key not in seen:
            seen.add(key)
            kept.append(row)

    return kept


def _exam_count(count: int) -> str:
    return _plural(count, "exame", "exames")


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _comparable_form(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c) and not c.isspace())
