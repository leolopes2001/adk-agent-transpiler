"""Semantic pass: rules that need the whole spec (graph, references, types, security).

Every check runs, even after others fail, and the diagnostics are returned
together; each check is written to tolerate graphs the others reject.
"""

from collections.abc import Callable
from pathlib import Path

from transpiler.errors import Diagnostic
from transpiler.semantic import budget, graph, refs, routes, types
from transpiler.semantic.context import SemanticContext
from transpiler.spec.models import AgentSpec

CHECKS: tuple[Callable[[SemanticContext], list[Diagnostic]], ...] = (
    graph.check,
    routes.check,
    types.check,
    refs.check,
    budget.check,
)


def validate(spec: AgentSpec, spec_dir: Path) -> list[Diagnostic]:
    ctx = SemanticContext.build(spec, spec_dir)
    return [diagnostic for check in CHECKS for diagnostic in check(ctx)]
