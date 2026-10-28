"""Passes collect every diagnostic before failing, so a spec is fixed in one round."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, TypeGuard

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from transpiler.spec.models import (
    ENV_VAR_PATTERN,
    IDENTIFIER_PATTERN,
    NODE_REF_PATTERN,
    ROUTE_PATTERN,
    JsonPath,
)

_INVALID_KEY_MARKER = "[key]"


class Code(StrEnum):
    E_IO = "E_IO"
    E_SPEC_TOO_LARGE = "E_SPEC_TOO_LARGE"
    E_SPEC_ENCODING = "E_SPEC_ENCODING"
    E_SPEC_JSON = "E_SPEC_JSON"
    E_DUPLICATE_KEY = "E_DUPLICATE_KEY"
    E_SCHEMA = "E_SCHEMA"
    E_MISSING_FIELD = "E_MISSING_FIELD"
    E_UNKNOWN_FIELD = "E_UNKNOWN_FIELD"
    E_UNKNOWN_KIND = "E_UNKNOWN_KIND"
    E_INVALID_IDENTIFIER = "E_INVALID_IDENTIFIER"
    E_RESERVED_IDENTIFIER = "E_RESERVED_IDENTIFIER"
    E_INVALID_ENV_NAME = "E_INVALID_ENV_NAME"
    E_THRESHOLD_ORDER = "E_THRESHOLD_ORDER"
    E_INLINE_SECRET = "E_INLINE_SECRET"  # noqa: S105 - a code, not a secret
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
    E_LLM_NOT_GUARDED = "E_LLM_NOT_GUARDED"
    E_SCHEDULE_NOT_GATED = "E_SCHEDULE_NOT_GATED"
    E_UNTRUSTED_AGENT_HAS_TOOLS = "E_UNTRUSTED_AGENT_HAS_TOOLS"
    E_AGENT_TOOLS_UNSUPPORTED = "E_AGENT_TOOLS_UNSUPPORTED"
    E_LLM_EXTRACT_REQUIRES_UNTRUSTED = "E_LLM_EXTRACT_REQUIRES_UNTRUSTED"
    E_PII_POLICY = "E_PII_POLICY"
    E_CODEGEN_RENDER = "E_CODEGEN_RENDER"
    E_CODEGEN_SYNTAX = "E_CODEGEN_SYNTAX"
    E_CODEGEN_FORBIDDEN = "E_CODEGEN_FORBIDDEN"
    E_CODEGEN_IMPORT = "E_CODEGEN_IMPORT"
    E_CODEGEN_TIMEOUT = "E_CODEGEN_TIMEOUT"
    E_CODEGEN_MISMATCH = "E_CODEGEN_MISMATCH"
    E_OUTPUT = "E_OUTPUT"
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


@dataclass(frozen=True, slots=True)
class _Located:
    path: JsonPath
    graph_node: Mapping[str, Any] | None


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


def errors_of(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.severity is Severity.ERROR]


def fail_on_errors(stage: Stage, diagnostics: Sequence[Diagnostic]) -> None:
    """Raises with every diagnostic, warnings included, if any of them is an error."""
    if errors_of(diagnostics):
        raise TranspileError(stage, diagnostics)


def json_pointer(path: Iterable[str | int]) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def diagnostics_from_validation_error(exc: ValidationError, raw: object) -> list[Diagnostic]:
    return [_diagnostic_from_error(error, raw) for error in exc.errors()]


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


def _diagnostic_from_error(error: ErrorDetails, raw: object) -> Diagnostic:
    located = _locate_in_raw(raw, error["loc"])
    node_id, kind = _node_label(located.graph_node)
    field = str(located.path[-1]) if located.path else ""
    message, hint = _message_and_hint(error, field, kind)
    return Diagnostic(
        code=_code_for(error),
        message=message,
        pointer=json_pointer(located.path),
        hint=hint,
        node_id=node_id,
        kind=kind,
    )


def _locate_in_raw(raw: object, loc: tuple[str | int, ...]) -> _Located:
    """Pydantic adds union tags and a "[key]" marker to `loc`; neither exists in the JSON."""
    path: list[str | int] = []
    current: object = raw
    graph_node: Mapping[str, Any] | None = None
    for part in loc:
        if part == _INVALID_KEY_MARKER:
            break

        if _is_union_tag(current, part):
            continue

        path.append(part)
        if isinstance(current, Mapping):
            current = current.get(str(part))
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
            if _is_graph_node(current, path):
                graph_node = current
        else:
            current = None

    return _Located(path=tuple(path), graph_node=graph_node)


def _is_union_tag(current: object, part: str | int) -> bool:
    return isinstance(current, Mapping) and part not in current and current.get("kind") == part


def _is_graph_node(value: object, path: list[str | int]) -> TypeGuard[Mapping[str, Any]]:
    parent = path[-2] if len(path) >= 2 else None  # noqa: PLR2004
    return parent == "nodes" and isinstance(value, Mapping)


def _node_label(node: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if node is None:
        return None, None

    node_id, kind = node.get("id"), node.get("kind")
    return (
        node_id if isinstance(node_id, str) else None,
        kind if isinstance(kind, str) else None,
    )


def _code_for(error: ErrorDetails) -> Code:
    if error["type"] == "string_pattern_mismatch":
        ctx = error.get("ctx") or {}
        return _PATTERN_CODES.get(str(ctx.get("pattern")), Code.E_SCHEMA)

    return _TYPE_CODES.get(error["type"], Code.E_SCHEMA)


def _message_and_hint(error: ErrorDetails, field: str, kind: str | None) -> tuple[str, str | None]:
    ctx = error.get("ctx") or {}
    match error["type"]:
        case "string_pattern_mismatch":
            return f"valor {error['input']!r} não segue o formato {ctx.get('pattern')}", None
        case "missing":
            message = f'{kind} requer "{field}"' if kind else f'campo obrigatório "{field}" ausente'
            return message, None
        case "union_tag_not_found":
            return 'campo obrigatório "kind" ausente', None
        case "extra_forbidden":
            where = f"em nós {kind}" if kind else "aqui"
            return f'campo "{field}" não é permitido {where}', None
        case "union_tag_invalid":
            return (
                f"kind {ctx.get('tag')!r} desconhecido",
                f"kinds válidos: {ctx.get('expected_tags')}",
            )
        case _:
            return error["msg"], None


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"
