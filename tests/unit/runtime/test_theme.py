"""The CLI's palette: colour only for success, failure and attention."""

import re
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from agent_runtime import terminal
from agent_runtime.nodes.present import ExamRow, RunResult, result_view, to_text
from agent_runtime.report import render_text
from agent_runtime.theme import THEME

pytestmark = pytest.mark.unit

SGR = re.compile(r"\x1b\[([0-9;]*)m")
GREEN, RED, YELLOW = "32", "31", "33"


def _colors(params: str) -> set[str]:
    """Foreground/background colour codes in one SGR sequence (attributes left out)."""
    codes = params.split(";")
    found: set[str] = set()
    i = 0
    while i < len(codes):
        code = codes[i]
        if code in {"38", "48"}:
            found.add(";".join(codes[i : i + (5 if codes[i + 1] == "2" else 3)]))
            i += 5 if codes[i + 1] == "2" else 3
            continue
        if code.isdigit() and (30 <= int(code) <= 49 or 90 <= int(code) <= 107):
            found.add(code)
        i += 1
    return found - {"39", "49"}


def _colors_by_line(text: str) -> list[tuple[str, set[str]]]:
    return [
        (line, set().union(*(_colors(m) for m in SGR.findall(line)))) for line in text.splitlines()
    ]


def _row(**fields: Any) -> ExamRow:
    base = {
        "mention": "x",
        "evidence": "x",
        "source": "both",
        "status": "rejected",
        "code": None,
        "name": None,
        "score": 0.5,
        "reason": "low_confidence",
    }
    return ExamRow.model_validate(base | fields)


ROWS = [
    _row(evidence="TSH", name="TSH", code="LAB-10043", status="accepted", reason=None),
    _row(evidence="PCR", name="Proteína C reativa", code="LAB-10037", status="accepted"),
    _row(evidence="T4", name="T4 total", code="LAB-10045", status="accepted", reason="reviewed"),
    _row(evidence="Curva insulinêmica", reason="already_accepted"),
    _row(evidence="T4 L", reason="needs_human_review"),
]


def _result(outcome: str, **fields: Any) -> RunResult:
    base = {
        "outcome": outcome,
        "format": "table",
        "exams": ROWS,
        "appointment": {"id": "apt_01JABCDEFGHJKMNPQRSTVWXYZ0", "status": "scheduled"},
        "reason": None,
        "user_message": None,
        "pii_masked": {"PERSON": 2, "CPF": 1},
        "injection_severity": "medium",
    }
    return RunResult.model_validate(base | fields)


RESULTS = [
    _result("scheduled"),
    _result("replayed"),
    _result("resolved", appointment=None),
    _result("not_scheduled", appointment=None, reason="patient_identifier_missing"),
    _result("schedule_failed", appointment=None, reason="Unknown exam code"),
    _result("rejected", appointment=None),
    *(
        _result(code, exams=[], appointment=None, user_message="Mensagem.")
        for code in ("OCR_LOW_QUALITY", "INPUT_BLOCKED", "NO_EXAMS_RESOLVED")
    ),
]
REPORT = {"run_id": "r1", "outcome": "error", "exit_code": 6, "timings_ms": {"total": 900}}


def _review_text(monkeypatch: pytest.MonkeyPatch) -> str:
    out = Console(file=StringIO(), width=100, force_terminal=True, theme=THEME)
    answers = iter(["1", "0"])

    def ask(prompt: str, *, choices: list[str], console: Console, **_: Any) -> str:
        console.print(terminal.Prompt(prompt, console=console, choices=choices).make_prompt(""))
        return next(answers)

    monkeypatch.setattr(terminal.Prompt, "ask", ask)
    payload = {
        "items": [
            {
                "index": 0,
                "mention": "Curva insulinêmica",
                "evidence": "Curva insulinêmica",
                "source": "both",
                "options": [{"code": "LAB-10015", "name": "TOTG", "score": 0.72}],
            },
            {"index": 1, "mention": "X", "evidence": "X", "source": "lexical", "options": []},
        ]
    }
    terminal.welcome(out, "agente", "faz coisas", "pedido.png")
    terminal.ask_review(payload, out)
    assert isinstance(out.file, StringIO)
    return out.file.getvalue()


def _outputs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return [
        _review_text(monkeypatch),
        *(to_text(result_view(result), color=True) for result in RESULTS),
        render_text({**REPORT, "path": [], "error_code": "MCP_TIMEOUT"}, None, color=True),
    ]


def test_only_green_red_and_yellow_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    used = set().union(*(c for text in _outputs(monkeypatch) for _, c in _colors_by_line(text)))

    assert used == {GREEN, RED, YELLOW}


def test_green_and_red_only_on_the_title_with_their_icon(monkeypatch: pytest.MonkeyPatch) -> None:
    for text in _outputs(monkeypatch):
        for line, colors in _colors_by_line(text):
            assert GREEN not in colors or "✔" in line
            assert RED not in colors or "✖" in line


def test_yellow_marks_differences_reviews_and_attention_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yellow = [
        SGR.sub("", line)
        for text in _outputs(monkeypatch)
        for line, colors in _colors_by_line(text)
        if YELLOW in colors
    ]

    assert yellow
    for line in yellow:
        assert any(mark in line for mark in ("PCR", "T4 ", "✎", "Atenção")), line


def test_the_review_has_no_colour_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    review = _review_text(monkeypatch)

    assert all(not colors for _, colors in _colors_by_line(review))
    assert "\x1b[1m" in review


def test_without_colour_there_are_no_colour_codes() -> None:
    for result in RESULTS:
        plain = to_text(result_view(result))
        assert "\x1b[" not in plain
