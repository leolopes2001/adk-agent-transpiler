"""Every diagnostic code has a spec file in tests/data/specs/invalid that triggers it.

Codes that a spec file cannot trigger by construction are listed in NOT_FROM_A_SPEC
with the reason; each of them must still be exercised by some other test.
"""

from pathlib import Path

import pytest

from transpiler.cli import main
from transpiler.errors import Code, Diagnostic, Severity, Stage, TranspileError
from transpiler.pipeline import compile_spec

pytestmark = pytest.mark.unit

TESTS = Path(__file__).parents[2]
INVALID = TESTS / "data" / "specs" / "invalid"

NOT_FROM_A_SPEC = {
    Code.E_IO: "a missing or unreadable file has no content to version",
    Code.E_SPEC_TOO_LARGE: "a file over 1 MiB is generated in tmp_path instead of versioned",
    Code.E_OUTPUT: "comes from the --out argument, not from the spec",
    Code.E_CODEGEN_RENDER: "a spec that passes the schema only yields renderable values",
    Code.E_CODEGEN_SYNTAX: "generated code from a valid spec always parses",
    Code.E_CODEGEN_FORBIDDEN: "the template cannot emit forbidden constructs",
    Code.E_CODEGEN_IMPORT: "needs a broken package or runtime, not a spec",
    Code.E_CODEGEN_TIMEOUT: "needs a package that hangs on import",
    Code.E_CODEGEN_MISMATCH: "needs generated code edited after rendering",
}
SCHEMA_CODES = {
    Code.E_SPEC_ENCODING,
    Code.E_SPEC_JSON,
    Code.E_DUPLICATE_KEY,
    Code.E_SCHEMA,
    Code.E_MISSING_FIELD,
    Code.E_UNKNOWN_FIELD,
    Code.E_UNKNOWN_KIND,
    Code.E_INVALID_IDENTIFIER,
    Code.E_RESERVED_IDENTIFIER,
    Code.E_INVALID_ENV_NAME,
    Code.E_THRESHOLD_ORDER,
    Code.E_INLINE_SECRET,
}
CORPUS_CODES = sorted(set(Code) - set(NOT_FROM_A_SPEC))


def _diagnostics(path: Path) -> tuple[Stage | None, tuple[Diagnostic, ...]]:
    try:
        compiled = compile_spec(path)
    except TranspileError as exc:
        return exc.stage, exc.diagnostics
    return None, compiled.warnings


def test_one_file_per_code() -> None:
    files = {path.stem for path in INVALID.glob("*.json")}

    assert files == {code.value for code in CORPUS_CODES}


@pytest.mark.parametrize("code", CORPUS_CODES, ids=str)
def test_file_triggers_its_code(code: Code) -> None:
    stage, diagnostics = _diagnostics(INVALID / f"{code}.json")

    assert code in {d.code for d in diagnostics}
    if code.startswith("W_"):
        assert stage is None
        assert all(d.severity is Severity.WARNING for d in diagnostics)
    else:
        assert stage is (Stage.SCHEMA if code in SCHEMA_CODES else Stage.SEMANTIC)


@pytest.mark.parametrize("code", CORPUS_CODES, ids=str)
def test_cli_exit_code(code: Code, capsys: pytest.CaptureFixture[str]) -> None:
    expected = 0 if code.startswith("W_") else 2 if code in SCHEMA_CODES else 3

    assert main(["validate", str(INVALID / f"{code}.json")]) == expected
    err = capsys.readouterr().err
    assert code.value in err
    if expected:
        assert err.rstrip().endswith("Nenhum arquivo foi gerado.")


def test_schema_files_point_where_expected() -> None:
    expected = {
        Code.E_SCHEMA: "/workflow/nodes/3/params/top_k",
        Code.E_MISSING_FIELD: "/workflow/nodes/0/server",
        Code.E_UNKNOWN_FIELD: "/workflow/nodes/1/color",
        Code.E_UNKNOWN_KIND: "/workflow/nodes/1",
        Code.E_INVALID_IDENTIFIER: "/workflow/nodes/1/id",
        Code.E_RESERVED_IDENTIFIER: "/workflow/nodes/1/id",
        Code.E_INVALID_ENV_NAME: "/mcp_servers/ocr/url_env",
        Code.E_THRESHOLD_ORDER: "/workflow/nodes/4/params",
        Code.E_INLINE_SECRET: "/http_services/scheduler/api_key",
    }
    for code, pointer in expected.items():
        _, diagnostics = _diagnostics(INVALID / f"{code}.json")
        assert (code, pointer) in {(d.code, d.pointer) for d in diagnostics}, code


@pytest.mark.parametrize("code", sorted(NOT_FROM_A_SPEC), ids=str)
def test_codes_without_a_file_are_tested_elsewhere(code: Code) -> None:
    sources = [
        path.read_text(encoding="utf-8")
        for path in TESTS.rglob("test_*.py")
        if path.name != Path(__file__).name
    ]

    assert any(code.value in source for source in sources), f"no test exercises {code}"


def test_every_diagnostic_code_is_in_the_corpus_or_exempt() -> None:
    assert set(CORPUS_CODES) | set(NOT_FROM_A_SPEC) == set(Code)
