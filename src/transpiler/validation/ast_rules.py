"""Allowlist, not denylist: generated modules are only imports and assignments of calls."""

import ast
from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic

_ALLOWED_NODES = (
    ast.Module,
    ast.ImportFrom,
    ast.alias,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Dict,
)
_ALLOWED_IMPORTS: dict[str, frozenset[str] | None] = {
    "__future__": frozenset({"annotations"}),
    "google.genai": frozenset({"types"}),
}
_ALLOWED_PREFIXES = ("google.adk", "agent_runtime")
_MODULE_NAMES = frozenset({"__all__"})


class _Checker:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bound: set[str] = set()

    def forbid(self, node: ast.AST, message: str) -> Diagnostic:
        return _forbidden_diagnostic(node, self.filename, message)

    def visit(self, node: ast.AST) -> Iterator[Diagnostic]:
        if not isinstance(node, _ALLOWED_NODES):
            yield self.forbid(node, f"construção não permitida: {type(node).__name__}")
        elif isinstance(node, ast.ImportFrom):
            yield from self.import_from(node)
        elif isinstance(node, ast.Assign):
            yield from self.assign(node)
        else:
            yield from self.expression(node)
            for child in ast.iter_child_nodes(node):
                yield from self.visit(child)

    def import_from(self, node: ast.ImportFrom) -> Iterator[Diagnostic]:
        if not _import_allowed(node):
            yield self.forbid(node, f"import não permitido: {'.' * node.level}{node.module or ''}")

        for alias in node.names:
            if alias.name == "*" or alias.asname is not None:
                yield self.forbid(node, "import com * ou alias não é permitido")
            self.bound.add(alias.asname or alias.name)

    def assign(self, node: ast.Assign) -> Iterator[Diagnostic]:
        yield from self.visit(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                yield self.forbid(target, "atribuição só a nomes simples")
                continue

            if target.id.startswith("_") and target.id not in _MODULE_NAMES:
                yield self.forbid(target, f"nome privado/dunder: {target.id}")
            self.bound.add(target.id)

    def expression(self, node: ast.AST) -> Iterator[Diagnostic]:
        if isinstance(node, ast.Name):
            yield from self._check_name(node)
        elif isinstance(node, ast.Attribute):
            yield from self._check_attribute(node)
        elif isinstance(node, ast.Call):
            yield from self._check_call(node)
        elif isinstance(node, ast.keyword):
            yield from self._check_keyword(node)
        elif isinstance(node, ast.Dict):
            yield from self._check_dict(node)

    def _check_name(self, node: ast.Name) -> Iterator[Diagnostic]:
        if node.id not in self.bound:
            yield self.forbid(node, f"nome não definido no módulo: {node.id}")

    def _check_attribute(self, node: ast.Attribute) -> Iterator[Diagnostic]:
        if node.attr.startswith("_"):
            yield self.forbid(node, f"atributo privado/dunder: {node.attr}")
        if not isinstance(node.value, ast.Name | ast.Attribute):
            yield self.forbid(node, "atributo só sobre nomes")

    def _check_call(self, node: ast.Call) -> Iterator[Diagnostic]:
        if not isinstance(node.func, ast.Name | ast.Attribute):
            yield self.forbid(node, "chamada só de nomes ou atributos")

    def _check_keyword(self, node: ast.keyword) -> Iterator[Diagnostic]:
        if node.arg is None:
            yield self.forbid(node, "**kwargs não é permitido")

    def _check_dict(self, node: ast.Dict) -> Iterator[Diagnostic]:
        if any(not (isinstance(k, ast.Constant) and isinstance(k.value, str)) for k in node.keys):
            yield self.forbid(node, "chaves de dict só como strings literais")


def check_tree(tree: ast.Module, filename: str) -> list[Diagnostic]:
    checker = _Checker(filename)
    return [d for statement in tree.body for d in checker.visit(statement)]


def check_source(source: str, filename: str) -> list[Diagnostic]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [
            Diagnostic(
                code=Code.E_CODEGEN_SYNTAX,
                message=f"{filename}:{exc.lineno}: {exc.msg}",
            )
        ]

    return check_tree(tree, filename)


def _forbidden_diagnostic(node: ast.AST, filename: str, message: str) -> Diagnostic:
    line = getattr(node, "lineno", 0)
    return Diagnostic(
        code=Code.E_CODEGEN_FORBIDDEN,
        message=f"{filename}:{line}: {message}",
    )


def _import_allowed(node: ast.ImportFrom) -> bool:
    if node.level:
        return node.level == 1 and node.module == "agent"

    module = node.module or ""
    if module in _ALLOWED_IMPORTS:
        allowed = _ALLOWED_IMPORTS[module]
        return allowed is None or all(alias.name in allowed for alias in node.names)

    return any(module == p or module.startswith(p + ".") for p in _ALLOWED_PREFIXES)
