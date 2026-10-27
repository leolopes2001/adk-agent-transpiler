"""What `lab-agent run` shows the operator on the terminal, besides the report.

The welcome header, the progress indicator and the review prompts go to
stderr, so stdout keeps only the report. Colour follows the CLI conventions:
none when `NO_COLOR` is set (and not empty) or when the stream is not a
terminal (a pipe, a file). Text that
comes from the document (evidence, mentions) or from the catalog is always
rendered as `Text`, never as rich markup: a document cannot style or fake
the prompt.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from typing import IO, Any

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from agent_runtime.nodes.human_review import NONE_CHOICE

NONE_LABEL = "nenhum destes — não agendar"
LABEL_WIDTH = 20

_READ_STYLE = "bold cyan"
_AI_STYLE = "bold magenta"
_CATALOG_STYLE = "bold green"


def color_enabled(stream: IO[str]) -> bool:
    """Colour only on a terminal, and never with NO_COLOR (https://no-color.org)."""
    if os.environ.get("NO_COLOR", ""):
        return False
    try:
        return stream.isatty()
    except ValueError:
        return False


def is_terminal(stream: IO[str]) -> bool:
    try:
        return stream.isatty()
    except ValueError:
        return False


def stderr_console() -> Console:
    """Console for what the operator reads on stderr; plain text off a terminal."""
    return Console(
        stderr=True,
        force_terminal=is_terminal(sys.stderr),
        no_color=not color_enabled(sys.stderr),
    )


class Progress:
    """Transient indicator on stderr naming the long step the run is in.

    `stages` are the workflow's long steps in order, as (name of the node
    whose output ends the step, label); the indicator shows the first one
    that is not over and disappears
    when it is stopped (before a review prompt and at the end of the run).
    """

    def __init__(self, console: Console, stages: Sequence[tuple[str, str]]) -> None:
        self._console = console
        self._pending = list(stages)
        self._status: Status | None = None
        self.shown: list[str] = []

    def _label(self) -> str | None:
        return f"{self._pending[0][1]}..." if self._pending else None

    def start(self) -> None:
        label = self._label()
        if label is None:
            self.stop()
            return
        if self._status is None:
            self._status = self._console.status(Text(label, style="dim"))
            self._status.start()
        else:
            self._status.update(Text(label, style="dim"))
        if not self.shown or self.shown[-1] != label:
            self.shown.append(label)

    def finished(self, node: str) -> None:
        """Output of `node`: the steps up to the one it ends are over."""
        names = [name for name, _ in self._pending]
        if node not in names:
            return
        del self._pending[: names.index(node) + 1]
        if self._status is not None:
            self.start()

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


def welcome(console: Console, name: str, description: str, image: str) -> None:
    """Who is running and on which file; printed before the run starts."""
    header = Text.assemble((name, "bold"), *((" — ", description) if description else ()))
    console.print(header)
    console.print(Text(f"Lendo {image}...", style="dim"))


def similarity(score: float) -> str:
    return f"{score:.2f}".replace(".", ",")


def review_intro(pending: int) -> Panel:
    noun = "item do pedido precisa" if pending == 1 else "itens do pedido precisam"
    body = Text.assemble(
        (f"{pending} {noun} da sua revisão.\n", "bold"),
        "Para cada um, escolha o exame ",
        ("EXATO", "bold"),
        " do catálogo que corresponde ao que está escrito no pedido.\n",
        ("0", "bold"),
        " = não agendar este item. ",
        ("Na dúvida, escolha 0.", "bold"),
    )
    return Panel(body, title="Revisão humana", title_align="left", border_style="yellow")


def _labelled(label: str, style: str, value: Text) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=LABEL_WIDTH, no_wrap=True)
    grid.add_column()
    grid.add_row(Text(label, style=style), value)
    return grid


def _interpretation(item: Mapping[str, Any]) -> Text:
    if item.get("source") == "lexical":
        return Text("— (item lido sem a IA)", style="dim")
    return Text(str(item.get("mention", "")), style="italic")


def _options(options: list[Mapping[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Exame")
    table.add_column("Código")
    table.add_column("Similaridade", justify="right")
    for number, option in enumerate(options, start=1):
        table.add_row(
            str(number),
            Text(str(option["name"])),
            Text(str(option["code"])),
            similarity(float(option["score"])),
        )
    table.add_row("0", Text(NONE_LABEL, style="yellow"), "", "")
    return table


def review_item(item: Mapping[str, Any], position: int, total: int) -> Group:
    """One review item: what the order says, what the model read, what the catalog offers."""
    options = list(item.get("options", []))
    return Group(
        Rule(Text(f"Item {position} de {total}", style="bold"), align="left"),
        _labelled("Lido no pedido", _READ_STYLE, Text(f"“{item.get('evidence', '')}”")),
        _labelled("Interpretação (IA)", _AI_STYLE, _interpretation(item)),
        Text("Opções do catálogo", style=_CATALOG_STYLE),
        Padding(_options(options), (0, 0, 0, 2)),
    )


def ask_review(payload: Mapping[str, Any], console: Console | None = None) -> dict[str, Any]:
    """Terminal prompt: one numbered choice per item, among the offered options only."""
    console = console or stderr_console()
    items = list(payload.get("items", []))
    choices: dict[str, str] = {}
    console.print(review_intro(len(items)))
    for position, item in enumerate(items, start=1):
        options = list(item.get("options", []))
        console.print()
        console.print(review_item(item, position, len(items)))
        answer = Prompt.ask(
            f"Sua escolha para o item {position}",
            choices=[str(n) for n in range(len(options) + 1)],
            console=console,
        )
        number = int(answer)
        choices[str(item["index"])] = options[number - 1]["code"] if number else NONE_CHOICE
    return {"choices": choices}
