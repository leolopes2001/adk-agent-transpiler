"""Structured diagnostics shared by every transpiler pass.

A diagnostic has a stable code (tests and users can rely on it), a JSON
Pointer into the spec and a message for humans. Passes collect all their
diagnostics before failing, so a user fixes a spec in one round instead of
one error per run.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from transpiler.spec.models import (
    ENV_VAR_PATTERN,
    IDENTIFIER_PATTERN,
    NODE_REF_PATTERN,
    ROUTE_PATTERN,
    JsonPath,
)


class Code(StrEnum):
    # load
    E_IO = "E_IO"
    E_SPEC_TOO_LARGE = "E_SPEC_TOO_LARGE"
    E_SPEC_ENCODING = "E_SPEC_ENCODING"
    E_SPEC_JSON = "E_SPEC_JSON"
    E_DUPLICATE_KEY = "E_DUPLICATE_KEY"
    # schema
    E_SCHEMA = "E_SCHEMA"
    E_MISSING_FIELD = "E_MISSING_FIELD"
    E_UNKNOWN_FIELD = "E_UNKNOWN_FIELD"
    E_UNKNOWN_KIND = "E_UNKNOWN_KIND"
    E_INVALID_IDENTIFIER = "E_INVALID_IDENTIFIER"
    E_RESERVED_IDENTIFIER = "E_RESERVED_IDENTIFIER"
    E_INVALID_ENV_NAME = "E_INVALID_ENV_NAME"
    E_THRESHOLD_ORDER = "E_THRESHOLD_ORDER"
    E_INLINE_SECRET = "E_INLINE_SECRET"  # noqa: S105 - a code, not a secret
    # semantic: graph
    E_DUPLICATE_ID = "E_DUPLICATE_ID"
    E_UNKNOWN_NODE = "E_UNKNOWN_NODE"
    E_NO_START = "E_NO_START"
    E_START_TARGET_INVALID = "E_START_TARGET_INVALID"
    E_UNREACHABLE = "E_UNREACHABLE"
    E_DEAD_END = "E_DEAD_END"
    E_TERMINAL_HAS_OUTGOING = "E_TERMINAL_HAS_OUTGOING"
    E_DUPLICATE_EDGE = "E_DUPLICATE_EDGE"
    E_FAN_IN_REQUIRES_JOIN = "E_FAN_IN_REQUIRES_JOIN"
    E_JOIN_ARITY = "E_JOIN_ARITY"
    E_JOIN_ROUTED_INPUT = "E_JOIN_ROUTED_INPUT"
    E_UNCONDITIONAL_CYCLE = "E_UNCONDITIONAL_CYCLE"
    E_SUBWORKFLOW_CYCLE = "E_SUBWORKFLOW_CYCLE"
    # semantic: routes, types, references, budget
    E_UNKNOWN_ROUTE = "E_UNKNOWN_ROUTE"
    E_ROUTE_NOT_COVERED = "E_ROUTE_NOT_COVERED"
    E_ROUTE_ON_NON_ROUTER = "E_ROUTE_ON_NON_ROUTER"
    E_MIXED_ROUTED_EDGES = "E_MIXED_ROUTED_EDGES"
    E_TYPE_MISMATCH = "E_TYPE_MISMATCH"
    E_GATE_CANDIDATES = "E_GATE_CANDIDATES"
    E_RECONCILE_WITHOUT_JOIN = "E_RECONCILE_WITHOUT_JOIN"
    E_UNKNOWN_REF = "E_UNKNOWN_REF"
    E_TOOL_NOT_ALLOWED = "E_TOOL_NOT_ALLOWED"
    E_TOOL_KIND_MISMATCH = "E_TOOL_KIND_MISMATCH"
    E_AGENT_REUSED = "E_AGENT_REUSED"
    E_TIMEOUT_BUDGET = "E_TIMEOUT_BUDGET"
    E_INSTRUCTION_FILE = "E_INSTRUCTION_FILE"
    # semantic: security
    E_LLM_NOT_GUARDED = "E_LLM_NOT_GUARDED"
    E_SCHEDULE_NOT_GATED = "E_SCHEDULE_NOT_GATED"
    E_UNTRUSTED_AGENT_HAS_TOOLS = "E_UNTRUSTED_AGENT_HAS_TOOLS"
    E_AGENT_TOOLS_UNSUPPORTED = "E_AGENT_TOOLS_UNSUPPORTED"
    E_LLM_EXTRACT_REQUIRES_UNTRUSTED = "E_LLM_EXTRACT_REQUIRES_UNTRUSTED"
    E_PII_POLICY = "E_PII_POLICY"
    # generated code
    E_CODEGEN_RENDER = "E_CODEGEN_RENDER"
    E_CODEGEN_SYNTAX = "E_CODEGEN_SYNTAX"
    E_CODEGEN_FORBIDDEN = "E_CODEGEN_FORBIDDEN"
    E_CODEGEN_IMPORT = "E_CODEGEN_IMPORT"
    E_CODEGEN_TIMEOUT = "E_CODEGEN_TIMEOUT"
    E_CODEGEN_MISMATCH = "E_CODEGEN_MISMATCH"
    # output
    E_OUTPUT = "E_OUTPUT"
    # warnings
    W_UNUSED_DECLARATION = "W_UNUSED_DECLARATION"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Stage(IntEnum):
    """Pass that failed; the value is the CLI exit code."""

    IO = 1
    SCHEMA = 2
    SEMANTIC = 3
    CODEGEN = 4


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: Code
    message: str
    pointer: str = ""
    hint: str | None = None
    node_id: str | None = None
    kind: str | None = None

    @property
    def severity(self) -> Severity:
        return Severity.WARNING if self.code.startswith("W_") else Severity.ERROR

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "pointer": self.pointer,
            "message": self.message,
            "hint": self.hint,
            "node_id": self.node_id,
            "kind": self.kind,
        }


class TranspileError(Exception):
    def __init__(self, stage: Stage, diagnostics: Sequence[Diagnostic]) -> None:
        self.stage = stage
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(f"{d.code}: {d.message}" for d in self.diagnostics))

    @property
    def exit_code(self) -> int:
        return self.stage.value


def errors_of(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.severity is Severity.ERROR]


def fail_on_errors(stage: Stage, diagnostics: Sequence[Diagnostic]) -> None:
    """Raises with every diagnostic (warnings included) if any of them is an error."""
    if errors_of(diagnostics):
        raise TranspileError(stage, diagnostics)


# --- JSON Pointer ---


def json_pointer(path: Iterable[str | int]) -> str:
    """RFC 6901 pointer for a path; `~` and `/` inside keys are escaped."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


# --- Pydantic translation ---

_PATTERN_CODES = {
    IDENTIFIER_PATTERN: Code.E_INVALID_IDENTIFIER,
    NODE_REF_PATTERN: Code.E_INVALID_IDENTIFIER,
    ROUTE_PATTERN: Code.E_INVALID_IDENTIFIER,
    ENV_VAR_PATTERN: Code.E_INVALID_ENV_NAME,
}
_TYPE_CODES = {
    "missing": Code.E_MISSING_FIELD,
    "union_tag_not_found": Code.E_MISSING_FIELD,
    "extra_forbidden": Code.E_UNKNOWN_FIELD,
    "union_tag_invalid": Code.E_UNKNOWN_KIND,
    "reserved_identifier": Code.E_RESERVED_IDENTIFIER,
    "threshold_order": Code.E_THRESHOLD_ORDER,
}


@dataclass(frozen=True, slots=True)
class _Located:
    path: JsonPath
    node: Mapping[str, Any] | None
    """The graph node the error is inside, if any."""


def _locate(raw: object, loc: tuple[str | int, ...]) -> _Located:
    """Maps a Pydantic `loc` onto the raw JSON.

    Pydantic inserts the discriminator value of tagged unions (e.g. the node
    kind) and a `[key]` marker for invalid dict keys; neither exists in the
    document, so both are dropped.
    """
    path: list[str | int] = []
    current: object = raw
    node: Mapping[str, Any] | None = None
    for part in loc:
        if part == "[key]":
            break
        if isinstance(current, Mapping):
            if part not in current and current.get("kind") == part:
                continue
            path.append(part)
            current = current.get(str(part))
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            path.append(part)
            current = current[part]
            parent = path[-2] if len(path) >= 2 else None  # noqa: PLR2004
            if parent == "nodes" and isinstance(current, Mapping):
                node = current
        else:
            path.append(part)
            current = None
    return _Located(path=tuple(path), node=node)


def _node_label(node: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if node is None:
        return None, None
    node_id, kind = node.get("id"), node.get("kind")
    return (
        node_id if isinstance(node_id, str) else None,
        kind if isinstance(kind, str) else None,
    )


def _translate(error: ErrorDetails, raw: object) -> Diagnostic:
    located = _locate(raw, error["loc"])
    node_id, kind = _node_label(located.node)
    field = str(located.path[-1]) if located.path else ""
    error_type = error["type"]
    ctx = error.get("ctx") or {}
    code = _TYPE_CODES.get(error_type, Code.E_SCHEMA)
    hint: str | None = None
    message = error["msg"]

    if error_type == "string_pattern_mismatch":
        code = _PATTERN_CODES.get(str(ctx.get("pattern")), Code.E_SCHEMA)
        message = f"valor {error['input']!r} não segue o formato {ctx.get('pattern')}"
    elif error_type == "missing":
        message = (
            f'{kind} requer "{field}"'
            if kind and located.node is not None
            else f'campo obrigatório "{field}" ausente'
        )
    elif error_type == "union_tag_not_found":
        message = 'campo obrigatório "kind" ausente'
    elif error_type == "extra_forbidden":
        where = f"em nós {kind}" if kind else "aqui"
        message = f'campo "{field}" não é permitido {where}'
    elif error_type == "union_tag_invalid":
        message = f"kind {ctx.get('tag')!r} desconhecido"
        hint = f"kinds válidos: {ctx.get('expected_tags')}"

    return Diagnostic(
        code=code,
        message=message,
        pointer=json_pointer(located.path),
        hint=hint,
        node_id=node_id,
        kind=kind,
    )


def diagnostics_from_validation_error(exc: ValidationError, raw: object) -> list[Diagnostic]:
    return [_translate(error, raw) for error in exc.errors()]


# --- rendering ---


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def summary(diagnostics: Sequence[Diagnostic]) -> str:
    errors = len(errors_of(diagnostics))
    warnings = len(diagnostics) - errors
    parts = [_plural(errors, "erro", "erros")]
    if warnings:
        parts.append(_plural(warnings, "aviso", "avisos"))
    return ", ".join(parts) + "."


def render_text(diagnostics: Sequence[Diagnostic]) -> str:
    width = max((len(d.code) for d in diagnostics), default=0)
    lines: list[str] = []
    for diag in diagnostics:
        icon = "✖" if diag.severity is Severity.ERROR else "⚠"
        header = f"{icon} {diag.code.value.ljust(width)}  {diag.pointer or '/'}"
        if diag.node_id:
            header += f'  (node "{diag.node_id}"' + (f", kind {diag.kind})" if diag.kind else ")")
        lines.append(header.rstrip())
        lines.append(f"  {diag.message}")
        if diag.hint:
            lines.append(f"  dica: {diag.hint}")
    return "\n".join(lines)


def render_json(diagnostics: Sequence[Diagnostic], extra: Mapping[str, Any] | None = None) -> str:
    errors = len(errors_of(diagnostics))
    document = {
        "ok": errors == 0,
        "errors": errors,
        "warnings": len(diagnostics) - errors,
        **(extra or {}),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }
    return json.dumps(document, ensure_ascii=False, indent=2)
