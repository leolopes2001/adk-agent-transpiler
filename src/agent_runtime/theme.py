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
