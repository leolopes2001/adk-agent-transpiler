"""AST allowlist for generated modules.

Generated code is a flat sequence of imports and assignments of calls, so the
rules allow exactly that shape instead of trying to list what is dangerous:
only allowlisted imports, only names bound earlier in the module (which rules
out every builtin, `exec` and `open` included), no private or dunder
attributes, and literals only as constants.
"""

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
# module -> allowed imported names (None: any public name)
_ALLOWED_IMPORTS: dict[str, frozenset[str] | None] = {
    "__future__": frozenset({"annotations"}),
    "google.genai": frozenset({"types"}),
}
_ALLOWED_PREFIXES = ("google.adk", "agent_runtime")
_MODULE_NAMES = frozenset({"__all__"})


def _forbidden(node: ast.AST, filename: str, message: str) -> Diagnostic:
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


class _Checker:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bound: set[str] = set()

    def forbid(self, node: ast.AST, message: str) -> Diagnostic:
        return _forbidden(node, self.filename, message)

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
        if isinstance(node, ast.Name) and node.id not in self.bound:
            yield self.forbid(node, f"nome não definido no módulo: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                yield self.forbid(node, f"atributo privado/dunder: {node.attr}")
            if not isinstance(node.value, ast.Name | ast.Attribute):
                yield self.forbid(node, "atributo só sobre nomes")
        elif isinstance(node, ast.Call) and not isinstance(node.func, ast.Name | ast.Attribute):
            yield self.forbid(node, "chamada só de nomes ou atributos")
        elif isinstance(node, ast.keyword) and node.arg is None:
            yield self.forbid(node, "**kwargs não é permitido")
        elif isinstance(node, ast.Dict) and any(
            not (isinstance(k, ast.Constant) and isinstance(k.value, str)) for k in node.keys
        ):
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
