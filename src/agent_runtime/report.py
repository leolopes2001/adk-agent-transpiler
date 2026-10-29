import time
from dataclasses import dataclass, field
from typing import Any

from google.adk import Event
from rich.console import Group, RenderableType
from rich.text import Text

from agent_runtime.messages import run_error
from agent_runtime.nodes.present import (
    REPORT_WIDTH,
    UNFINISHED_TITLE,
    ExamRow,
    RunResult,
    details_view,
    result_view,
    summary,
    to_text,
)
from agent_runtime.theme import FAIL, MUTED

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_BLOCKED = 3
EXIT_LOW_QUALITY = 4
EXIT_NO_EXAMS = 5
EXIT_DEPENDENCY = 6
EXIT_STALE_CODE = 7
EXIT_CONFIG = 8

EXIT_BY_OUTCOME = {
    "scheduled": EXIT_OK,
    "replayed": EXIT_OK,
    "resolved": EXIT_OK,
    # A run that ends without an appointment request must not look like success.
    "not_scheduled": EXIT_DEPENDENCY,
    "schedule_failed": EXIT_DEPENDENCY,
    "rejected": EXIT_NO_EXAMS,
    "OCR_LOW_QUALITY": EXIT_LOW_QUALITY,
    "INPUT_BLOCKED": EXIT_BLOCKED,
    "NO_EXAMS_RESOLVED": EXIT_NO_EXAMS,
}


# Keeps names, routes, timestamps and error codes only: no node output reaches the report.
@dataclass
class RunTrace:
    started: float = field(default_factory=time.monotonic)
    # Wall clock, the clock of Event.timestamp.
    started_at: float = field(default_factory=time.time)
    path: list[str] = field(default_factory=list)
    first_seen: dict[str, float] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    routes: dict[str, Any] = field(default_factory=dict)
    finished: float | None = None

    def observe(self, event: Event) -> None:
        name = top_level_name(event)
        if name is None:
            return

        if name not in self.first_seen:
            self.path.append(name)
            self.first_seen[name] = event.timestamp
        self.last_seen[name] = event.timestamp
        if event.error_code:
            node = event.node_info.name or name
            self.failures[node] = self.failures.get(node, 0) + 1
        if event.actions.route is not None and event.node_info.name == name:
            self.routes[name] = event.actions.route

    def finish(self) -> None:
        self.finished = time.monotonic()

    def timings_ms(self) -> dict[str, int]:
        timings: dict[str, int] = {}
        previous_end = self.started_at
        for name in self.path:
            end = self.last_seen[name]
            timings[name] = max(0, round((end - previous_end) * 1000))
            previous_end = max(previous_end, end)

        end = self.finished if self.finished is not None else time.monotonic()
        timings["total"] = round((end - self.started) * 1000)
        return timings


def top_level_name(event: Event) -> str | None:
    parts = [p for p in event.node_info.path.split("/") if p]
    return parts[1].split("@", 1)[0] if len(parts) > 1 else None


def exit_code_for(result: RunResult) -> int:
    return EXIT_BY_OUTCOME[result.outcome]


def build_report(  # noqa: PLR0913 - every field of the report
    *,
    run_id: str,
    agent: str,
    spec_sha256: str | None,
    trace: RunTrace,
    exit_code: int,
    result: RunResult | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_id": run_id,
        "agent": agent,
        "spec_sha256": spec_sha256,
        "outcome": result.outcome if result else "error",
        "exit_code": exit_code,
        "path": list(trace.path),
        "timings_ms": trace.timings_ms(),
        "retries": dict(trace.failures),
        **_result_fields(result),
    }
    if result and result.user_message:
        report["user_message"] = result.user_message
    if error_code:
        report["error_code"] = error_code

    return report


def error_message(error_code: str, node: str | None = None) -> tuple[str, str]:
    what, action = run_error(error_code, node)
    return what[:1].upper() + what[1:] + ".", action


def render_text(
    report: dict[str, Any],
    result: RunResult | None,
    *,
    verbose: bool = False,
    width: int = REPORT_WIDTH,
    color: bool = False,
) -> str:
    parts: list[RenderableType] = []
    if result:
        parts.append(details_view(result) if verbose else result_view(result))
    elif report.get("error_code"):
        path = report.get("path") or []
        parts.append(_error_view(report["error_code"], path[-1] if path else None))

    parts += [Text(), Text(_footer(report, verbose=verbose), style=MUTED)]
    return to_text(Group(*parts), width, color=color)


def _result_fields(result: RunResult | None) -> dict[str, Any]:
    if result is None:
        return {"pii_masked": {}, "injection_severity": None, "exams": [], "appointment": None}

    return {
        "pii_masked": dict(result.pii_masked),
        "injection_severity": result.injection_severity,
        "exams": [_exam_entry(row) for row in result.exams],
        "appointment": result.appointment.model_dump() if result.appointment else None,
    }


def _exam_entry(row: ExamRow) -> dict[str, Any]:
    return {
        "mention": row.mention,
        "code": row.code,
        "score": row.score,
        "status": row.status,
        "source": row.source,
        "reason": row.reason,
    }


def _error_view(error_code: str, node: str | None) -> Group:
    message, action = error_message(error_code, node)
    return summary(
        UNFINISHED_TITLE,
        FAIL,
        [
            ("Motivo", message),
            ("O que fazer", action),
            ("Código", Text(error_code, style=MUTED)),
        ],
    )


def _footer(report: dict[str, Any], *, verbose: bool) -> str:
    total_ms = report["timings_ms"]["total"]
    if verbose:
        return (
            f"Execução {report['run_id']}: {report['outcome']} (exit {report['exit_code']}) "
            f"em {total_ms} ms; caminho: {' → '.join(report['path']) or '—'}"
        )

    return f"Execução {report['run_id']} · {_seconds_label(total_ms)}"


def _seconds_label(ms: int) -> str:
    return f"{ms / 1000:.1f} s".replace(".", ",")
