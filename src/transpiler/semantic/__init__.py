"""Every check runs even when others fail, so each must tolerate graphs the others reject."""

from collections.abc import Callable
from pathlib import Path

from transpiler.errors import Diagnostic
from transpiler.semantic import budget, graph, refs, routes, security, types
from transpiler.semantic.context import SemanticContext
from transpiler.spec.models import AgentSpec

CHECKS: tuple[Callable[[SemanticContext], list[Diagnostic]], ...] = (
    graph.check,
    routes.check,
    types.check,
    refs.check,
    budget.check,
    security.check,
)


def validate(spec: AgentSpec, spec_dir: Path) -> list[Diagnostic]:
    ctx = SemanticContext.build(spec, spec_dir)
    return [diagnostic for check in CHECKS for diagnostic in check(ctx)]
