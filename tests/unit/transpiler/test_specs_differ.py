"""The two versioned specs must produce different programs, not one template with new names."""

import ast
from pathlib import Path

import pytest

from transpiler.codegen.render import render
from transpiler.pipeline import compile_spec

pytestmark = pytest.mark.unit

SPECS = Path(__file__).parents[3] / "agent_specs"
MAIN = SPECS / "exam_scheduling_agent.json"
LOOKUP = SPECS / "examples" / "exam_lookup_only.json"


def _module(spec: Path) -> ast.Module:
    return ast.parse(render(compile_spec(spec).program, spec_name=spec.name).agent_py)


def _imports(module: ast.Module) -> set[str]:
    return {
        f"{node.module}.{alias.name}"
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _calls(module: ast.Module) -> dict[str, str]:
    """Variable name -> callee, for every top-level `name = callee(...)`."""
    result: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            [target] = node.targets
            assert isinstance(target, ast.Name)
            result[target.id] = ast.unparse(node.value.func)
    return result


def _edges(module: ast.Module, workflow: str) -> set[str]:
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == workflow
            and isinstance(node.value, ast.Call)
        ):
            edges = next(kw.value for kw in node.value.keywords if kw.arg == "edges")
            assert isinstance(edges, ast.List)
            return {ast.unparse(element) for element in edges.elts}
    raise AssertionError(f"workflow {workflow} not found")


@pytest.fixture(scope="module")
def main_module() -> ast.Module:
    return _module(MAIN)


@pytest.fixture(scope="module")
def lookup_module() -> ast.Module:
    return _module(LOOKUP)


def test_both_specs_are_valid() -> None:
    assert compile_spec(MAIN).warnings == ()
    assert compile_spec(LOOKUP).warnings == ()


def test_imports_differ(main_module: ast.Module, lookup_module: ast.Module) -> None:
    main, lookup = _imports(main_module), _imports(lookup_module)

    assert main - lookup == {
        "google.adk.Agent",
        "google.adk.apps.ResumabilityConfig",
        "google.adk.workflow.JoinNode",
        "google.genai.types",
        "agent_runtime.guardrails",
        "agent_runtime.schemas",
        "agent_runtime.models.resolve_model",
        "agent_runtime.bootstrap.HttpServiceConfig",
    }
    assert lookup <= main


def test_nodes_differ(main_module: ast.Module, lookup_module: ast.Module) -> None:
    main, lookup = _calls(main_module), _calls(lookup_module)

    assert set(main) - set(lookup) == {
        "llm_extract_input",
        "exam_extractor",
        "merge",
        "reconcile",
        "review",
        "schedule",
        "extraction",
    }
    assert set(lookup) - set(main) == set()
    assert main["exam_extractor"] == "Agent"
    assert main["merge"] == "JoinNode"
    assert "Agent" not in lookup.values()
    assert "JoinNode" not in lookup.values()


def test_edges_differ(main_module: ast.Module, lookup_module: ast.Module) -> None:
    main, lookup = _edges(main_module, "root_agent"), _edges(lookup_module, "root_agent")

    assert "(guard, {'clean': lexical_extract, 'blocked': abort})" in lookup
    assert "(gate, {'accepted': present, 'review': present, 'empty': abort})" in lookup
    assert "(guard, {'clean': extraction, 'blocked': abort})" in main
    assert main.isdisjoint({"(lexical_extract, resolve)"})
    assert main != lookup


def test_only_main_spec_is_resumable(main_module: ast.Module, lookup_module: ast.Module) -> None:
    assert "resumability_config" in ast.unparse(main_module)
    assert "resumability_config" not in ast.unparse(lookup_module)


def test_lookup_gate_has_no_review_band() -> None:
    program = compile_spec(LOOKUP).program
    gate = next(node for node in program.leaf_nodes() if node.var == "gate")

    assert gate.kwargs["review_threshold"] == gate.kwargs["accept_threshold"]
