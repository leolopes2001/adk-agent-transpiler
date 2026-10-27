"""Strict reading of spec files and of the instruction files they reference.

`json.load` alone is too lenient for an input that drives code generation: it
keeps the last of duplicated keys, accepts NaN and reads files of any size.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from transpiler.spec.models import AgentSpec, find_inline_secrets

MAX_SPEC_BYTES = 1024 * 1024
MAX_INSTRUCTION_BYTES = 32 * 1024


class TranspileError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


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


def _read_limited(path: Path, limit: int, code: str, what: str) -> bytes:
    try:
        if not path.is_file():
            raise TranspileError(code, f"{what} não encontrado ou não é um arquivo: {path}")
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as exc:
        raise TranspileError(code, f"não foi possível ler {what} {path}: {exc.strerror}") from exc
    if len(data) > limit:
        raise TranspileError(code, f"{what} excede o limite de {limit} bytes: {path}")
    return data


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TranspileError("E_DUPLICATE_KEY", f'chave "{key}" repetida no mesmo objeto')
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise TranspileError("E_SPEC_JSON", f"{name} não é um valor JSON válido")


def parse_json(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranspileError("E_SPEC_ENCODING", f"a spec não é UTF-8 válido: {exc}") from exc
    try:
        raw = json.loads(
            text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise TranspileError(
            "E_SPEC_JSON", f"JSON inválido na linha {exc.lineno}, coluna {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise TranspileError("E_SPEC_JSON", "a spec deve ser um objeto JSON")
    return raw


def parse_spec(raw: dict[str, Any]) -> AgentSpec:
    secrets = find_inline_secrets(raw)
    if secrets:
        keys = ", ".join("/".join(str(part) for part in path) for path in secrets)
        raise TranspileError("E_INLINE_SECRET", f"segredo literal na spec: {keys}")
    try:
        return AgentSpec.model_validate(raw)
    except ValidationError as exc:
        raise TranspileError("E_SCHEMA", str(exc)) from exc


def load_spec(path: Path) -> LoadedSpec:
    data = _read_limited(path, MAX_SPEC_BYTES, "E_IO", "arquivo de spec")
    spec = parse_spec(parse_json(data))
    return LoadedSpec(path=path, sha256=hashlib.sha256(data).hexdigest(), spec=spec)


def read_instruction(spec_dir: Path, relative_path: str) -> InstructionFile:
    """Reads an instruction file, which must live under the spec's directory."""
    if Path(relative_path).is_absolute():
        raise TranspileError(
            "E_INSTRUCTION_FILE", f"instruction_file deve ser relativo: {relative_path}"
        )
    base = spec_dir.resolve()
    target = (base / relative_path).resolve()
    if not target.is_relative_to(base):
        raise TranspileError(
            "E_INSTRUCTION_FILE",
            f"instruction_file aponta para fora do diretório da spec: {relative_path}",
        )
    data = _read_limited(target, MAX_INSTRUCTION_BYTES, "E_INSTRUCTION_FILE", "instruction_file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranspileError(
            "E_INSTRUCTION_FILE", f"instruction_file não é UTF-8 válido: {relative_path}"
        ) from exc
    return InstructionFile(
        relative_path=relative_path, text=text, sha256=hashlib.sha256(data).hexdigest()
    )
