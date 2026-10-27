"""What `lab-agent run` shows the operator on the terminal, besides the report.

The welcome header and the review prompts go to stderr, so stdout keeps only the report. Text that
comes from the document (evidence, mentions) or from the catalog is always
rendered as `Text`, never as rich markup: a document cannot style or fake
the prompt.
"""

from collections.abc import Mapping
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent_runtime.nodes.human_review import NONE_CHOICE

NONE_LABEL = "nenhum destes — não agendar"
LABEL_WIDTH = 20

_READ_STYLE = "bold cyan"
_AI_STYLE = "bold magenta"
_CATALOG_STYLE = "bold green"


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
    console = console or Console(stderr=True)
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
