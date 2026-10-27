"""The one colour palette of `lab-agent run` on the terminal.

Text, borders, rules and headers keep the terminal's default colour; headers
and labels are bold. Secondary information is dim. Colour carries meaning
and only on the icon or the key word: green for success, red for failure,
yellow for differences and attention. Every console of the CLI uses `THEME`,
which turns rich's own coloured defaults (prompt choices, rules, spinner,
table titles) into the same palette. With NO_COLOR or off a terminal the
colours go away and the text, icons and bold still say everything.
"""

from rich.theme import Theme

LABEL = "bold"
MUTED = "dim"
OK = "bold green"
FAIL = "bold red"
WARN = "yellow"

ICON_OK = "✔"
ICON_FAIL = "✖"
ICON_REVIEWED = "✎"

THEME = Theme(
    {
        "prompt.choices": LABEL,
        "prompt.default": LABEL,
        "prompt.invalid": "none",
        "prompt.invalid.choice": "none",
        "rule.line": "none",
        "rule.text": "none",
        "status.spinner": MUTED,
        "table.title": LABEL,
        "table.header": LABEL,
        "table.caption": MUTED,
        "table.border": "none",
    }
)
