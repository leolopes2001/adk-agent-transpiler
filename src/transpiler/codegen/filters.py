"""Jinja filters that are the only way spec values reach generated source.

`pyrepr` turns already-validated primitives into Python literals and refuses
anything else; `pyident` re-checks names before they become variables. The
template never interpolates a spec value without one of them, so a spec
cannot inject code even if an earlier pass let something through.
"""

import ast
import keyword
import math
import re
from collections.abc import Mapping

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ESCAPES = {"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


class CodegenValueError(ValueError):
    """A value that cannot be emitted safely."""


def _string(value: str) -> str:
    # Same quote choice as ruff format, so the output is already formatted.
    quote = "'" if value.count('"') > value.count("'") else '"'
    parts: list[str] = []
    for char in value:
        if char == quote:
            parts.append("\\" + char)
        elif char in _ESCAPES:
            parts.append(_ESCAPES[char])
        elif not char.isprintable():
            parts.append(repr(char)[1:-1])
        else:
            parts.append(char)
    literal = quote + "".join(parts) + quote
    if ast.literal_eval(literal) != value:  # pragma: no cover - guards the escaping above
        raise CodegenValueError(f"string escaping failed for {value!r}")
    return literal


def pyrepr(value: object) -> str:
    """Python literal for str, int, float, bool, None and tuples/mappings of them."""
    if value is None or isinstance(value, bool | int):
        return repr(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodegenValueError(f"non-finite float: {value!r}")
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, tuple):
        items = [pyrepr(item) for item in value]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CodegenValueError("mapping keys must be strings")
        return "{" + ", ".join(f"{pyrepr(k)}: {pyrepr(v)}" for k, v in value.items()) + "}"
    raise CodegenValueError(f"type {type(value).__name__} cannot be emitted as a literal")


def pyident(value: object) -> str:
    """The value itself, if it is a plain Python identifier that is not a keyword."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or keyword.iskeyword(value):
        raise CodegenValueError(f"not a safe identifier: {value!r}")
    return value


def string_lines(value: str) -> list[str]:
    """Literals for each line of a text."""
    return [_string(line) for line in value.splitlines(keepends=True)] or ['""']


def text_assignment(name: str, value: str, line_length: int = 100) -> str:
    """`name = <text>`, one literal per line of text, laid out the way ruff format would.

    Ruff drops the parentheses around a single literal and joins implicitly
    concatenated parts that fit on one line; matching that keeps the generated
    file stable under `ruff format`.
    """
    name = pyident(name)
    parts = string_lines(value)
    single = f"{name} = {_string(value)}"
    if len(parts) == 1 or len(single) <= line_length:
        return single
    body = "\n".join(f"    {part}" for part in parts)
    return f"{name} = (\n{body}\n)"
