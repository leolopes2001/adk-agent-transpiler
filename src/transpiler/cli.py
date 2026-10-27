"""`lab-transpile`: validate specs, generate ADK packages and print the JSON Schema.

Exit codes: 0 ok, 1 usage or I/O, 2 schema, 3 semantic, 4 generated-code validation.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from transpiler.errors import Diagnostic, Stage, TranspileError, render_json, render_text, summary
from transpiler.pipeline import compile_spec, generate
from transpiler.spec.schema_export import render_schema


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(Stage.IO.value, f"{self.prog}: erro: {message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="lab-transpile", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="valida a spec sem gerar arquivos")
    validate.add_argument("spec", type=Path)
    validate.add_argument("--format", choices=("text", "json"), default="text")

    generate = commands.add_parser("generate", help="gera o pacote ADK a partir da spec")
    generate.add_argument("spec", type=Path)
    generate.add_argument("--out", type=Path, default=Path("generated"))
    generate.add_argument("--format", choices=("text", "json"), default="text")

    commands.add_parser("schema", help="imprime o JSON Schema das specs")
    return parser


def _relative(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def _report_failure(error: TranspileError, output_format: str) -> int:
    if output_format == "json":
        sys.stdout.write(render_json(error.diagnostics) + "\n")
    else:
        sys.stderr.write(render_text(error.diagnostics) + "\n")
        sys.stderr.write(f"{summary(error.diagnostics)} Nenhum arquivo foi gerado.\n")
    return error.exit_code


def _report_warnings(warnings: Sequence[Diagnostic]) -> None:
    if warnings:
        sys.stderr.write(render_text(warnings) + "\n")


def _validate(spec: Path, output_format: str) -> int:
    compiled = compile_spec(spec)
    if output_format == "json":
        sys.stdout.write(render_json(compiled.warnings, {"spec": str(spec)}) + "\n")
    else:
        _report_warnings(compiled.warnings)
        notes = f" ({len(compiled.warnings)} aviso(s))" if compiled.warnings else ""
        sys.stdout.write(f"✔ spec válida: {spec}{notes}\n")
    return 0


def _generate(spec: Path, out: Path, output_format: str) -> int:
    result = generate(spec, out, allowed_root=Path.cwd())
    warnings = result.compiled.warnings
    if output_format == "json":
        extra = {"spec": str(spec), "output": str(result.path), "files": list(result.files)}
        sys.stdout.write(render_json(warnings, extra) + "\n")
    else:
        _report_warnings(warnings)
        files = ", ".join(result.files)
        sys.stdout.write(f"✔ gerado: {_relative(result.path)} ({files})\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "schema":
        sys.stdout.write(render_schema())
        return 0
    try:
        if args.command == "validate":
            return _validate(args.spec, args.format)
        return _generate(args.spec, args.out, args.format)
    except TranspileError as error:
        return _report_failure(error, args.format)


if __name__ == "__main__":
    raise SystemExit(main())
