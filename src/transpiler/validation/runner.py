"""Runs every check on a generated package and compares the probe with the lowered program.

Order matters: the AST rules and compilation run on the text before anything
is imported, so the probe never executes code that failed the allowlist.
"""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transpiler.errors import Code, Diagnostic, Stage, TranspileError, fail_on_errors
from transpiler.lowering import AdkGraph, AdkProgram
from transpiler.validation.ast_rules import check_source
from transpiler.validation.compile_check import compile_source

PROBE_TIMEOUT_S = 30.0
PACKAGE_FILES = ("agent.py", "__init__.py")
_DUMMY_URL = "http://invalid.local"
_OUTPUT_TAIL = 2000


@dataclass(frozen=True, slots=True)
class ProbeReport:
    data: Mapping[str, Any]

    @property
    def adk_version(self) -> str:
        return str(self.data["adk_version"])


def dummy_env(program: AdkProgram) -> dict[str, str]:
    """Placeholder values for every env var the program names; no real secret reaches the probe."""
    env: dict[str, str] = {}
    for server in program.servers:
        env[server.url_env] = f"{_DUMMY_URL}/{server.key}/sse"
        env[server.auth_token_env] = "probe-dummy"
    for service in program.services:
        env[service.base_url_env] = f"{_DUMMY_URL}/{service.key}"
        env[service.api_key_env] = "probe-dummy"
    return env


def _edges(graph: AdkGraph) -> list[list[str | None]]:
    edges: list[list[str | None]] = []
    for edge in graph.edges:
        if edge.routes:
            edges.extend([edge.source, target, route] for route, target in edge.routes)
        else:
            edges.append([edge.source, edge.target, None])
    return sorted(edges, key=lambda e: tuple(part or "" for part in e))


def expected_workflows(program: AdkProgram) -> dict[str, list[list[str | None]]]:
    workflows = {program.agent_name: _edges(program.root)}
    for node in program.subworkflows():
        if node.subgraph is not None:
            workflows[node.var] = _edges(node.subgraph)
    return workflows


def _codegen_error(code: Code, message: str) -> TranspileError:
    return TranspileError(Stage.CODEGEN, [Diagnostic(code=code, message=message)])


def check_files(package_dir: Path) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for name in PACKAGE_FILES:
        source = (package_dir / name).read_text(encoding="utf-8")
        found = check_source(source, name)
        diagnostics.extend(found or compile_source(source, name))
    return diagnostics


def run_probe(
    package_dir: Path, env: Mapping[str, str], timeout: float = PROBE_TIMEOUT_S
) -> dict[str, Any]:
    # -B: no __pycache__ inside the package that is about to be emitted.
    command = [sys.executable, "-I", "-B", "-m", "transpiler.validation.probe", str(package_dir)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv; the package passed the AST rules
            command,
            env={"PATH": os.defpath, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _codegen_error(
            Code.E_CODEGEN_TIMEOUT, f"o import do pacote gerado excedeu {timeout:g}s"
        ) from exc
    if completed.returncode != 0:
        tail = completed.stderr.strip()[-_OUTPUT_TAIL:]
        raise _codegen_error(
            Code.E_CODEGEN_IMPORT,
            f"o pacote gerado não importa (exit {completed.returncode}):\n{tail}",
        )
    try:
        result: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise _codegen_error(Code.E_CODEGEN_IMPORT, "o probe não produziu JSON") from exc
    return result


def compare(result: Mapping[str, Any], program: AdkProgram) -> list[Diagnostic]:
    expected: dict[str, Any] = {
        "root_agent_is_workflow": True,
        "root_agent_name": program.agent_name,
        "app_is_app": True,
        "app_resumable": program.needs_resumability,
        "exports": ["app", "root_agent"],
        "node_names": program.node_names(),
        "variables_match_names": True,
        "workflows": expected_workflows(program),
        "network_attempts": [],
    }
    return [
        Diagnostic(
            code=Code.E_CODEGEN_MISMATCH,
            message=f"{key}: esperado {value!r}, o módulo gerado tem {result.get(key)!r}",
        )
        for key, value in expected.items()
        if result.get(key) != value
    ]


def validate_package(
    package_dir: Path, program: AdkProgram, *, timeout: float = PROBE_TIMEOUT_S
) -> ProbeReport:
    """Raises TranspileError (stage CODEGEN) on the first failing step."""
    fail_on_errors(Stage.CODEGEN, check_files(package_dir))
    result = run_probe(package_dir, dummy_env(program), timeout)
    fail_on_errors(Stage.CODEGEN, compare(result, program))
    return ProbeReport(result)
