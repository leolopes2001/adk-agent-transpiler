import os
import sys
from collections.abc import Mapping, Sequence
from typing import IO, Any

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import InvalidResponse, Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from agent_runtime.nodes.human_review import NONE_CHOICE
from agent_runtime.theme import LABEL, MUTED, THEME

NONE_LABEL = "nenhum destes — não agendar"
LABEL_WIDTH = 21


class Progress:
    def __init__(self, console: Console, stages: Sequence[tuple[str, str]]) -> None:
        self._console = console
        self._pending = list(stages)
        self._status: Status | None = None
        self.shown: list[str] = []

    def start(self) -> None:
        label = self._current_label()
        if label is None:
            self.stop()
            return

        if self._status is None:
            self._status = self._console.status(Text(label, style=MUTED))
            self._status.start()
        else:
            self._status.update(Text(label, style=MUTED))

        if not self.shown or self.shown[-1] != label:
            self.shown.append(label)

    def finished(self, node: str) -> None:
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

    def _current_label(self) -> str | None:
        return f"{self._pending[0][1]}..." if self._pending else None


# rich's own message for an invalid answer is in English.
class ChoicePrompt(Prompt):
    def process_response(self, value: str) -> str:
        answer = value.strip()
        if self.choices is not None and answer not in self.choices:
            raise InvalidResponse(
                Text(f"Opção inválida. Digite um dos números: {spoken_list(self.choices)}.")
            )

        return answer


def color_enabled(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR", ""):
        return False

    return is_terminal(stream)


def is_terminal(stream: IO[str]) -> bool:
    try:
        return stream.isatty()
    except ValueError:
        return False


def stderr_console() -> Console:
    return Console(
        stderr=True,
        force_terminal=is_terminal(sys.stderr),
        no_color=not color_enabled(sys.stderr),
        theme=THEME,
        highlight=False,
    )


def welcome(console: Console, name: str, description: str, image: str) -> None:
    header = Text.assemble((name, LABEL), *((" — ", description) if description else ()))
    console.print(header)
    console.print(Text(f"Lendo {image}...", style=MUTED))


def spoken_list(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} ou {items[-1]}"


def similarity(score: float) -> str:
    return f"{score:.2f}".replace(".", ",")


def review_intro(pending: int) -> Panel:
    noun = "item do pedido precisa" if pending == 1 else "itens do pedido precisam"
    body = Text.assemble(
        (f"{pending} {noun} da sua revisão.\n", LABEL),
        "Para cada um, escolha o exame ",
        ("EXATO", LABEL),
        " do catálogo que corresponde ao que está escrito no pedido.\n",
        ("0", LABEL),
        " = não agendar este item. ",
        ("Na dúvida, escolha 0.", LABEL),
    )
    return Panel(
        body, title=Text("Revisão humana", style=LABEL), title_align="left", border_style="none"
    )


def review_item(item: Mapping[str, Any], position: int, total: int) -> Group:
    options = list(item.get("options", []))
    return Group(
        Rule(Text(f"Item {position} de {total}", style=LABEL), align="left", style="none"),
        _labelled("Lido no pedido", Text(f"“{item.get('evidence', '')}”")),
        _labelled("Interpretação (IA)", _interpretation(item)),
        Text(),
        Text("Opções do catálogo", style=LABEL),
        Padding(_options(options), (0, 0, 0, 2)),
    )


def ask_review(payload: Mapping[str, Any], console: Console | None = None) -> dict[str, Any]:
    console = console or stderr_console()
    items = list(payload.get("items", []))
    choices: dict[str, str] = {}
    console.print(review_intro(len(items)))
    for position, item in enumerate(items, start=1):
        options = list(item.get("options", []))
        console.print()
        console.print(review_item(item, position, len(items)))
        answer = ChoicePrompt.ask(
            f"Sua escolha para o item {position}",
            choices=[str(n) for n in range(len(options) + 1)],
            console=console,
        )
        number = int(answer)
        choices[str(item["index"])] = options[number - 1]["code"] if number else NONE_CHOICE

    return {"choices": choices}


def _labelled(label: str, value: Text) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=LABEL_WIDTH, no_wrap=True)
    grid.add_column()
    grid.add_row(Text(label, style=LABEL), value)
    return grid


def _interpretation(item: Mapping[str, Any]) -> Text:
    if item.get("source") == "lexical":
        return Text("— (item lido sem a IA)", style=MUTED)
    return Text(f"“{item.get('mention', '')}”")


def _options(options: list[Mapping[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, header_style=LABEL)
    table.add_column("#", justify="right")
    table.add_column("Exame")
    table.add_column("Código")
    table.add_column("Similaridade", justify="right")
    for number, option in enumerate(options, start=1):
        table.add_row(
            str(number),
            Text(str(option["name"])),
            Text(str(option["code"])),
            Text(similarity(float(option["score"])), style=MUTED),
        )
    table.add_row("0", Text(NONE_LABEL), "", "")
    return table
