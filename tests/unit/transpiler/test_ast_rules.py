from pathlib import Path
from typing import Any

import pytest

from transpiler.codegen.render import RenderedPackage, render
from transpiler.errors import Code
from transpiler.ir.builder import build_program
from transpiler.lowering import AdkProgram, lower
from transpiler.spec.loader import InstructionFile, parse_spec
from transpiler.validation.ast_rules import check_source
from transpiler.validation.compile_check import compile_source
from transpiler.validation.runner import check_files, compare, dummy_env, expected_workflows

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]

INSTRUCTION = InstructionFile("instructions/extractor.md", "Extraia exames.\n", "ab" * 32)
PRELUDE = "from agent_runtime import nodes\n"


@pytest.fixture
def adk(spec_dict: SpecDict) -> AdkProgram:
    program = build_program(
        parse_spec(spec_dict),
        spec_sha256="0" * 64,
        instructions={name: INSTRUCTION for name in spec_dict["llm_agents"]},
    )
    return lower(program)


@pytest.fixture
def package(adk: AdkProgram) -> RenderedPackage:
    return render(adk, spec_name="spec.json")


def test_generated_sources_pass(package: RenderedPackage) -> None:
    assert check_source(package.agent_py, "agent.py") == []
    assert check_source(package.init_py, "__init__.py") == []


@pytest.mark.parametrize(
    "snippet",
    [
        "import os",
        "from os import system",
        "from subprocess import run",
        "from google.genai import Client",
        "from google.adk import *",
        "from agent_runtime import nodes as n",
        "from . import agent",
        "from ..agent import app",
        "x = exec('1')",
        "x = eval('1')",
        "x = open('/etc/passwd')",
        "x = __import__('os')",
        "x = getattr(nodes, 'abort')",
        "x = compile('1', 'f', 'exec')",
        "x = print",
        "x = nodes.__class__",
        "x = nodes._pending('a')",
        "x = nodes.abort.__globals__",
        "x = lambda: 1",
        "def f():\n    pass",
        "class C:\n    pass",
        "global nodes",
        "nodes.abort(name='x')",
        "x = nodes.abort(**{'name': 'x'})",
        "x = (1, 2)[0]",
        "x = f'{nodes}'",
        "x = [n for n in (1,)]",
        "x = 1 + 1",
        "x, y = 1, 2",
        "nodes.x = 1",
        "__builtins__ = 1",
        "x = {nodes: 1}",
        "x = nodes.abort(name='a')(1)",
        "x = undefined_name",
        "with nodes:\n    pass",
        "if True:\n    x = 1",
    ],
)
def test_forbidden_constructs(snippet: str) -> None:
    diagnostics = check_source(PRELUDE + snippet + "\n", "agent.py")

    assert diagnostics
    assert {d.code for d in diagnostics} == {Code.E_CODEGEN_FORBIDDEN}
    assert diagnostics[0].message.startswith("agent.py:")


def test_names_must_be_bound_before_use() -> None:
    source = "from agent_runtime import nodes\nx = y\ny = nodes.abort(name='y')\n"

    [diagnostic] = check_source(source, "agent.py")
    assert "y" in diagnostic.message


def test_syntax_error() -> None:
    [diagnostic] = check_source("x = (\n", "agent.py")

    assert diagnostic.code is Code.E_CODEGEN_SYNTAX


def test_compile_check() -> None:
    assert compile_source("x = 1\n", "agent.py") == []
    [diagnostic] = compile_source("x = 1\0\n", "agent.py")
    assert diagnostic.code is Code.E_CODEGEN_SYNTAX


def test_check_files(tmp_path: Path, package: RenderedPackage) -> None:
    (tmp_path / "agent.py").write_text(package.agent_py, encoding="utf-8")
    (tmp_path / "__init__.py").write_text("import os\n", encoding="utf-8")

    [diagnostic] = check_files(tmp_path)
    assert diagnostic.message.startswith("__init__.py:1:")


def test_dummy_env_covers_every_variable(adk: AdkProgram) -> None:
    env = dummy_env(adk)

    assert set(env) == {
        "OCR_MCP_URL",
        "RAG_MCP_URL",
        "MCP_AUTH_TOKEN",
        "SCHEDULER_API_URL",
        "SCHEDULER_API_KEY",
    }
    assert all("invalid.local" in env[k] for k in ("OCR_MCP_URL", "SCHEDULER_API_URL"))


def test_expected_workflows(adk: AdkProgram) -> None:
    workflows = expected_workflows(adk)

    assert set(workflows) == {"full_agent", "extraction"}
    assert ["gate", "abort", "empty"] in workflows["full_agent"]
    assert ["merge", "reconcile", None] in workflows["extraction"]


def _probe_result(adk: AdkProgram) -> dict[str, Any]:
    return {
        "root_agent_is_workflow": True,
        "root_agent_name": "full_agent",
        "app_is_app": True,
        "app_resumable": True,
        "exports": ["app", "root_agent"],
        "node_names": adk.node_names(),
        "variables_match_names": True,
        "workflows": expected_workflows(adk),
        "network_attempts": [],
    }


def test_compare_accepts_matching_probe(adk: AdkProgram) -> None:
    assert compare(_probe_result(adk), adk) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("root_agent_is_workflow", False),
        ("root_agent_name", "other"),
        ("app_resumable", False),
        ("node_names", ["ocr"]),
        ("network_attempts", ["('example.com', 80)"]),
    ],
)
def test_compare_reports_mismatch(adk: AdkProgram, key: str, value: object) -> None:
    result = {**_probe_result(adk), key: value}

    [diagnostic] = compare(result, adk)
    assert diagnostic.code is Code.E_CODEGEN_MISMATCH
    assert diagnostic.message.startswith(f"{key}:")
