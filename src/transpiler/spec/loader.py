"""Strict reading of spec files and of the instruction files they reference.

`json.load` alone is too lenient for an input that drives code generation: it
keeps the last of duplicated keys, accepts NaN and reads files of any size.
Nesting is limited too (`MAX_SPEC_DEPTH`; the specs use 6 levels): a small
file of deeply nested lists would otherwise exhaust the recursion limit of
the parser or of a later pass and crash with a traceback (audit F06).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from transpiler.errors import (
    Code,
    Diagnostic,
    Stage,
    TranspileError,
    diagnostics_from_validation_error,
    json_pointer,
)
from transpiler.spec.models import AgentSpec, find_inline_secrets

MAX_SPEC_BYTES = 1024 * 1024
MAX_INSTRUCTION_BYTES = 32 * 1024
MAX_SPEC_DEPTH = 32
_TOO_DEEP = f"a spec excede {MAX_SPEC_DEPTH} níveis de aninhamento"


@dataclass(frozen=True, slots=True)
class LoadedSpec:
    path: Path
    sha256: str
    spec: AgentSpec


@dataclass(frozen=True, slots=True)
class InstructionFile:
    relative_path: str
    text: str
    sha256: str


def _fail(stage: Stage, code: Code, message: str, pointer: str = "") -> TranspileError:
    return TranspileError(stage, [Diagnostic(code=code, message=message, pointer=pointer)])


def read_spec_bytes(path: Path) -> bytes:
    try:
        if not path.is_file():
            raise _fail(Stage.IO, Code.E_IO, f"arquivo de spec não encontrado: {path}")
        with path.open("rb") as handle:
            data = handle.read(MAX_SPEC_BYTES + 1)
    except OSError as exc:
        raise _fail(Stage.IO, Code.E_IO, f"não foi possível ler {path}: {exc.strerror}") from exc
    if len(data) > MAX_SPEC_BYTES:
        raise _fail(
            Stage.SCHEMA, Code.E_SPEC_TOO_LARGE, f"a spec excede {MAX_SPEC_BYTES} bytes: {path}"
        )
    return data


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(Stage.SCHEMA, Code.E_DUPLICATE_KEY, f'chave "{key}" repetida no objeto')
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise _fail(Stage.SCHEMA, Code.E_SPEC_JSON, f"{name} não é um valor JSON válido")


def parse_json(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _fail(
            Stage.SCHEMA, Code.E_SPEC_ENCODING, f"a spec não é UTF-8 válido (byte {exc.start})"
        ) from exc
    try:
        raw = json.loads(
            text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise _fail(
            Stage.SCHEMA,
            Code.E_SPEC_JSON,
            f"JSON inválido na linha {exc.lineno}, coluna {exc.colno}: {exc.msg}",
        ) from exc
    except RecursionError:
        raise _fail(Stage.SCHEMA, Code.E_SCHEMA, _TOO_DEEP) from None
    if not isinstance(raw, dict):
        raise _fail(Stage.SCHEMA, Code.E_SPEC_JSON, "a spec deve ser um objeto JSON")
    return raw


def _check_depth(raw: object) -> None:
    """Fails on the first container nested deeper than MAX_SPEC_DEPTH (no recursion)."""
    pending: list[tuple[object, tuple[str | int, ...]]] = [(raw, ())]
    while pending:
        value, path = pending.pop()
        if isinstance(value, dict):
            children: list[tuple[str | int, object]] = list(value.items())
        elif isinstance(value, list):
            children = list(enumerate(value))
        else:
            continue
        if len(path) >= MAX_SPEC_DEPTH:
            raise _fail(Stage.SCHEMA, Code.E_SCHEMA, _TOO_DEEP, json_pointer(path))
        pending.extend((child, (*path, key)) for key, child in children)


def parse_spec(raw: dict[str, Any]) -> AgentSpec:
    """Schema pass: inline-secret scan plus model validation, reporting every error found."""
    _check_depth(raw)
    diagnostics = [
        Diagnostic(
            code=Code.E_INLINE_SECRET,
            message=f'"{path[-1]}" parece um segredo com valor literal',
            pointer=json_pointer(path),
            hint=f'use "{path[-1]}_env" com o nome de uma variável de ambiente',
        )
        for path in find_inline_secrets(raw)
    ]
    try:
        spec = AgentSpec.model_validate(raw)
    except ValidationError as exc:
        secret_pointers = {d.pointer for d in diagnostics}
        diagnostics.extend(
            d
            for d in diagnostics_from_validation_error(exc, raw)
            if not (d.code is Code.E_UNKNOWN_FIELD and d.pointer in secret_pointers)
        )
        raise TranspileError(Stage.SCHEMA, diagnostics) from exc
    if diagnostics:
        raise TranspileError(Stage.SCHEMA, diagnostics)
    return spec


def load_spec(path: Path) -> LoadedSpec:
    data = read_spec_bytes(path)
    spec = parse_spec(parse_json(data))
    return LoadedSpec(path=path, sha256=hashlib.sha256(data).hexdigest(), spec=spec)


def read_instruction(spec_dir: Path, relative_path: str, pointer: str = "") -> InstructionFile:
    """Reads an instruction file, which must live under the spec's directory."""

    def fail(message: str) -> TranspileError:
        return _fail(Stage.SEMANTIC, Code.E_INSTRUCTION_FILE, message, pointer)

    if Path(relative_path).is_absolute():
        raise fail(f"instruction_file deve ser relativo: {relative_path}")
    base = spec_dir.resolve()
    target = (base / relative_path).resolve()
    if not target.is_relative_to(base):
        raise fail(f"instruction_file aponta para fora do diretório da spec: {relative_path}")
    try:
        if not target.is_file():
            raise fail(f"instruction_file não encontrado: {relative_path}")
        with target.open("rb") as handle:
            data = handle.read(MAX_INSTRUCTION_BYTES + 1)
    except OSError as exc:
        raise fail(f"não foi possível ler {relative_path}: {exc.strerror}") from exc
    if len(data) > MAX_INSTRUCTION_BYTES:
        raise fail(f"instruction_file excede {MAX_INSTRUCTION_BYTES} bytes: {relative_path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise fail(f"instruction_file não é UTF-8 válido: {relative_path}") from exc
    return InstructionFile(
        relative_path=relative_path, text=text, sha256=hashlib.sha256(data).hexdigest()
    )
