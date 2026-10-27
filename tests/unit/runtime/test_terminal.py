import re
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from agent_runtime import cli, terminal

pytestmark = pytest.mark.unit

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAYLOAD: dict[str, Any] = {
    "items": [
        {
            "index": 1,
            "mention": "Tiroxina livre",
            "evidence": "2. T4",
            "source": "both",
            "options": [
                {"code": "LAB-10044", "name": "T4 livre", "score": 0.8},
                {"code": "LAB-10045", "name": "T4 total", "score": 0.7963},
            ],
        },
        {
            "index": 3,
            "mention": "X",
            "evidence": "[bold]X[/bold]",
            "source": "lexical",
            "options": [],
        },
    ]
}


def _console(**kwargs: Any) -> Console:
    return Console(file=StringIO(), width=100, **kwargs)


def _answer(monkeypatch: pytest.MonkeyPatch, *answers: str) -> list[list[str]]:
    offered: list[list[str]] = []
    pending = iter(answers)

    def ask(*_: Any, choices: list[str], **__: Any) -> str:
        offered.append(choices)
        return next(pending)

    monkeypatch.setattr(terminal.Prompt, "ask", ask)
    return offered


def test_ask_review_maps_numbers_to_offered_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    offered = _answer(monkeypatch, "2", "0")

    assert terminal.ask_review(PAYLOAD, _console()) == {"choices": {"1": "LAB-10045", "3": "none"}}
    assert offered == [["0", "1", "2"], ["0"]]
    assert cli.ask_review is terminal.ask_review


@pytest.mark.parametrize(
    "console",
    [
        pytest.param({"color_system": None}, id="no-color"),
        pytest.param({"force_terminal": True, "no_color": True}, id="tty-no-color"),
    ],
)
def test_review_explains_itself_without_color(
    monkeypatch: pytest.MonkeyPatch, console: dict[str, Any]
) -> None:
    _answer(monkeypatch, "1", "0")
    out = _console(**console)

    terminal.ask_review(PAYLOAD, out)

    assert isinstance(out.file, StringIO)
    text = ANSI.sub("", out.file.getvalue())
    for expected in (
        "Revisão humana",
        "2 itens do pedido precisam da sua revisão",
        "EXATO",
        "0 = não agendar este item",
        "Na dúvida, escolha 0",
        "Item 1 de 2",
        "Item 2 de 2",
        "Lido no pedido",
        "“2. T4”",
        "Interpretação (IA)",
        "Interpretação (IA)    “Tiroxina livre”",
        "Opções do catálogo",
        "Similaridade",
        "T4 livre",
        "LAB-10044",
        "0,80",
        "nenhum destes — não agendar",
        "item lido sem a IA",
    ):
        assert expected in text
    assert "score" not in text.lower()


def test_text_from_the_document_is_not_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    _answer(monkeypatch, "1", "0")
    out = _console(color_system=None)

    terminal.ask_review(PAYLOAD, out)

    assert isinstance(out.file, StringIO)
    assert "“[bold]X[/bold]”" in out.file.getvalue()


def test_intro_for_a_single_item() -> None:
    out = _console(color_system=None)

    out.print(terminal.review_intro(1))

    assert isinstance(out.file, StringIO)
    assert "1 item do pedido precisa da sua revisão" in out.file.getvalue()


def test_interpretation_and_options_are_separated_by_a_blank_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer(monkeypatch, "1", "0")
    out = _console(color_system=None)

    terminal.ask_review(PAYLOAD, out)

    assert isinstance(out.file, StringIO)
    lines = [line.rstrip() for line in out.file.getvalue().splitlines()]
    at = lines.index("Interpretação (IA)    “Tiroxina livre”")
    assert lines[at + 1 : at + 3] == ["", "Opções do catálogo"]


MAGENTA = re.compile(
    r"\x1b\[(?:[0-9;]*;)?(35|95|38;5;(5|13|90|91|12[5-9]|13[0-5]|16[2-9]|17[01]|20[01]))(?:;[0-9;]*)?m"
)


def test_no_magenta_anywhere_in_the_review(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _console(force_terminal=True, color_system="truecolor", theme=terminal.THEME)
    answers = iter(["1", "0"])

    def ask(prompt: str, *, choices: list[str], console: Console, **_: Any) -> str:
        console.print(terminal.Prompt(prompt, console=console, choices=choices).make_prompt(""))
        return next(answers)

    monkeypatch.setattr(terminal.Prompt, "ask", ask)

    terminal.ask_review(PAYLOAD, out)

    assert isinstance(out.file, StringIO)
    text = out.file.getvalue()
    assert "[0/1/2]" in ANSI.sub("", text)
    assert not MAGENTA.search(text)
    assert "\x1b[" in text


def test_the_review_console_uses_the_blue_choices_theme() -> None:
    console = terminal.stderr_console()

    assert str(console.get_style("prompt.choices")) == "bold cyan"
