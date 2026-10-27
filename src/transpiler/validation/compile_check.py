"""Byte-compilation of generated sources, after the AST rules passed."""

from transpiler.errors import Code, Diagnostic


def compile_source(source: str, filename: str) -> list[Diagnostic]:
    try:
        compile(source, filename, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        return [Diagnostic(code=Code.E_CODEGEN_SYNTAX, message=f"{filename}: {exc}")]
    return []
