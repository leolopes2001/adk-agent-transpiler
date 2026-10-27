"""Run report and exit codes of `lab-agent run`.

`RunTrace` follows the ADK events of a run and keeps only names, routes,
timestamps and error codes; never outputs. The report combines it with the
`RunResult` the terminal node wrote, which is already free of personal data
(counts per PII type, masked mentions, catalog codes).
"""

import time
from dataclasses import dataclass, field
from typing import Any

from google.adk import Event

from agent_runtime.nodes.present import RunResult, render_table

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


def top_level_name(event: Event) -> str | None:
    """Root-workflow node an event belongs to ("<root>@1/extraction@1/merge@1" -> extraction)."""
    parts = [p for p in event.node_info.path.split("/") if p]
    return parts[1].split("@", 1)[0] if len(parts) > 1 else None


@dataclass
class RunTrace:
    started: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.time)
    """Wall clock, the same clock as `Event.timestamp`."""
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
        """Per node: from the previous node's last event (or the run start) to its last event."""
        timings: dict[str, int] = {}
        previous_end = self.started_at
        for name in self.path:
            end = self.last_seen[name]
            timings[name] = max(0, round((end - previous_end) * 1000))
            previous_end = max(previous_end, end)
        end = self.finished if self.finished is not None else time.monotonic()
        timings["total"] = round((end - self.started) * 1000)
        return timings


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
        "pii_masked": dict(result.pii_masked) if result else {},
        "injection_severity": result.injection_severity if result else None,
        "exams": [
            {
                "mention": row.mention,
                "code": row.code,
                "score": row.score,
                "status": row.status,
                "source": row.source,
                "reason": row.reason,
            }
            for row in (result.exams if result else [])
        ],
        "appointment": result.appointment.model_dump() if result and result.appointment else None,
    }
    if result and result.user_message:
        report["user_message"] = result.user_message
    if error_code:
        report["error_code"] = error_code
    return report


def render_text(report: dict[str, Any], result: RunResult | None) -> str:
    """Human-readable report: the result table plus one line about the run."""
    lines = [render_table(result).rstrip()] if result else []
    timings = report["timings_ms"]
    lines.append(
        f"Execução {report['run_id']}: {report['outcome']} (exit {report['exit_code']}) "
        f"em {timings['total']} ms; caminho: {' → '.join(report['path']) or '—'}"
    )
    return "\n".join(lines) + "\n"
