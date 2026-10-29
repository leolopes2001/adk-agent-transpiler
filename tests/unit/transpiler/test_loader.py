import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from transpiler.errors import Stage, TranspileError
from transpiler.spec.loader import (
    MAX_INSTRUCTION_BYTES,
    MAX_SPEC_BYTES,
    load_spec,
    parse_json,
    read_instruction,
)

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]


def _code(exc: pytest.ExceptionInfo[TranspileError]) -> str:
    [diagnostic] = exc.value.diagnostics
    return diagnostic.code


def test_load_valid_spec(spec_dict: SpecDict, write_spec: Any) -> None:
    path = write_spec(spec_dict)

    loaded = load_spec(path)

    assert loaded.spec.agent.name == "full_agent"
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TranspileError) as exc:
        load_spec(tmp_path / "nope.json")
    assert _code(exc) == "E_IO"
    assert exc.value.exit_code == 1


def test_directory_is_not_a_spec(tmp_path: Path) -> None:
    with pytest.raises(TranspileError) as exc:
        load_spec(tmp_path)
    assert _code(exc) == "E_IO"


def test_spec_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_bytes(b" " * (MAX_SPEC_BYTES + 1))

    with pytest.raises(TranspileError) as exc:
        load_spec(path)
    assert _code(exc) == "E_SPEC_TOO_LARGE"
    assert exc.value.stage is Stage.SCHEMA


def test_not_utf8() -> None:
    with pytest.raises(TranspileError) as exc:
        parse_json(b'{"a": "\xff"}')
    assert _code(exc) == "E_SPEC_ENCODING"


@pytest.mark.parametrize(
    "data",
    [b"{", b'{"a": 1,}', b"", b'{"a": NaN}', b'{"a": Infinity}', b'{"a": ' + b"9" * 5000 + b"}"],
)
def test_invalid_json(data: bytes) -> None:
    with pytest.raises(TranspileError) as exc:
        parse_json(data)
    assert _code(exc) == "E_SPEC_JSON"


def test_top_level_must_be_object() -> None:
    with pytest.raises(TranspileError) as exc:
        parse_json(b"[]")
    assert _code(exc) == "E_SPEC_JSON"


@pytest.mark.parametrize("data", [b'{"a": 1, "a": 2}', b'{"x": {"b": 1, "b": 1}}'])
def test_duplicate_keys(data: bytes) -> None:
    with pytest.raises(TranspileError) as exc:
        parse_json(data)
    assert _code(exc) == "E_DUPLICATE_KEY"


def test_inline_secret_is_rejected(spec_dict: SpecDict, write_spec: Any) -> None:
    spec_dict["http_services"]["scheduler"]["api_key"] = "abc123"

    with pytest.raises(TranspileError) as exc:
        load_spec(write_spec(spec_dict))
    assert _code(exc) == "E_INLINE_SECRET"
    assert "abc123" not in str(exc.value)


def test_schema_error(spec_dict: SpecDict, write_spec: Any) -> None:
    spec_dict["spec_version"] = "2.0"

    with pytest.raises(TranspileError) as exc:
        load_spec(write_spec(spec_dict))
    assert _code(exc) == "E_SCHEMA"


def test_instruction_error_carries_pointer(tmp_path: Path) -> None:
    with pytest.raises(TranspileError) as exc:
        read_instruction(tmp_path, "nope.md", pointer="/llm_agents/a/instruction_file")
    assert exc.value.diagnostics[0].pointer == "/llm_agents/a/instruction_file"
    assert exc.value.stage is Stage.SEMANTIC


def test_read_instruction(tmp_path: Path) -> None:
    (tmp_path / "instructions").mkdir()
    (tmp_path / "instructions" / "a.md").write_text("Olá, instrução.\n", encoding="utf-8")

    instruction = read_instruction(tmp_path, "instructions/a.md")

    assert instruction.text == "Olá, instrução.\n"
    assert instruction.sha256 == hashlib.sha256("Olá, instrução.\n".encode()).hexdigest()


@pytest.mark.parametrize("relative", ["../outside.md", "instructions/../../outside.md"])
def test_instruction_cannot_escape_spec_dir(tmp_path: Path, relative: str) -> None:
    spec_dir = tmp_path / "spec"
    (spec_dir / "instructions").mkdir(parents=True)
    (tmp_path / "outside.md").write_text("x", encoding="utf-8")

    with pytest.raises(TranspileError) as exc:
        read_instruction(spec_dir, relative)
    assert _code(exc) == "E_INSTRUCTION_FILE"


def test_instruction_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(TranspileError) as exc:
        read_instruction(tmp_path, str(target))
    assert _code(exc) == "E_INSTRUCTION_FILE"


def test_instruction_symlink_escape(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (tmp_path / "secret.md").write_text("x", encoding="utf-8")
    os.symlink(tmp_path / "secret.md", spec_dir / "link.md")

    with pytest.raises(TranspileError) as exc:
        read_instruction(spec_dir, "link.md")
    assert _code(exc) == "E_INSTRUCTION_FILE"


def test_instruction_missing(tmp_path: Path) -> None:
    with pytest.raises(TranspileError) as exc:
        read_instruction(tmp_path, "nope.md")
    assert _code(exc) == "E_INSTRUCTION_FILE"


def test_instruction_size_limit(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_bytes(b"a" * (MAX_INSTRUCTION_BYTES + 1))

    with pytest.raises(TranspileError) as exc:
        read_instruction(tmp_path, "big.md")
    assert _code(exc) == "E_INSTRUCTION_FILE"


def test_instruction_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe")

    with pytest.raises(TranspileError) as exc:
        read_instruction(tmp_path, "bad.md")
    assert _code(exc) == "E_INSTRUCTION_FILE"
