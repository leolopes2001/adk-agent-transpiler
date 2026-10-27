import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

from transpiler.codegen.render import render
from transpiler.errors import Code, TranspileError
from transpiler.ir.builder import build_program
from transpiler.lowering import AdkProgram, lower
from transpiler.spec.loader import load_spec, read_instruction
from transpiler.validation import probe
from transpiler.validation.runner import run_probe, validate_package

pytestmark = pytest.mark.integration

SPEC = Path(__file__).parents[1] / "data" / "specs" / "valid" / "full.json"


def _program() -> AdkProgram:
    loaded = load_spec(SPEC)
    instructions = {
        key: read_instruction(SPEC.parent, agent.instruction_file)
        for key, agent in loaded.spec.llm_agents.items()
    }
    return lower(build_program(loaded.spec, spec_sha256=loaded.sha256, instructions=instructions))


@pytest.fixture
def generated(tmp_path: Path) -> tuple[Path, AdkProgram]:
    program = _program()
    package = render(program, spec_name=SPEC.name)
    target = tmp_path / program.agent_name
    target.mkdir()
    (target / "agent.py").write_text(package.agent_py, encoding="utf-8")
    (target / "__init__.py").write_text(package.init_py, encoding="utf-8")
    return target, program


def _write_package(root: Path, name: str, agent_source: str) -> Path:
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text("from .agent import app, root_agent\n", encoding="utf-8")
    (package / "agent.py").write_text(agent_source, encoding="utf-8")
    return package


def test_generated_package_imports_without_network(generated: tuple[Path, AdkProgram]) -> None:
    package_dir, program = generated

    report = validate_package(package_dir, program)

    assert report.data["root_agent_type"].startswith("google.adk")
    assert report.data["root_agent_type"].endswith(".Workflow")
    assert report.data["node_names"] == program.node_names()
    assert report.data["network_attempts"] == []
    assert report.adk_version == "2.13.0"


def test_probe_detects_edited_graph(generated: tuple[Path, AdkProgram]) -> None:
    package_dir, program = generated
    agent = package_dir / "agent.py"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace("        (schedule, present),\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(TranspileError) as exc:
        validate_package(package_dir, program)
    assert [d.code for d in exc.value.diagnostics] == [Code.E_CODEGEN_MISMATCH]
    assert exc.value.diagnostics[0].message.startswith("workflows:")


def test_ast_rules_run_before_import(generated: tuple[Path, AdkProgram]) -> None:
    package_dir, program = generated
    marker = package_dir.parent / "executed"
    (package_dir / "__init__.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).touch()\n", encoding="utf-8"
    )

    with pytest.raises(TranspileError) as exc:
        validate_package(package_dir, program)
    assert exc.value.diagnostics[0].code is Code.E_CODEGEN_FORBIDDEN
    assert not marker.exists()


def test_probe_blocks_network(tmp_path: Path) -> None:
    source = (
        "import socket\n"
        "socket.create_connection(('example.com', 80), timeout=1)\n"
        "app = root_agent = None\n"
    )
    package = _write_package(tmp_path, "net_agent", source)

    with pytest.raises(TranspileError) as exc:
        run_probe(package, env={})
    [diagnostic] = exc.value.diagnostics
    assert diagnostic.code is Code.E_CODEGEN_IMPORT
    assert "network access is blocked" in diagnostic.message


def test_probe_timeout(tmp_path: Path) -> None:
    package = _write_package(tmp_path, "slow_agent", "import time\ntime.sleep(30)\n")

    with pytest.raises(TranspileError) as exc:
        run_probe(package, env={}, timeout=2)
    assert exc.value.diagnostics[0].code is Code.E_CODEGEN_TIMEOUT


def test_probe_runs_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PYTHONPATH must not leak into the probe (python -I), so a shadowing module is ignored.
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "json.py").write_text("raise SystemExit('shadowed')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    program = _program()
    package = render(program, spec_name=SPEC.name)
    target = _write_package(tmp_path, program.agent_name, package.agent_py)
    (target / "__init__.py").write_text(package.init_py, encoding="utf-8")

    assert run_probe(target, env={"OCR_MCP_URL": "x"})["root_agent_name"] == program.agent_name


@pytest.mark.filterwarnings("ignore:.*EXPERIMENTAL.*ResumabilityConfig:UserWarning")
def test_inspect_package_in_process(
    generated: tuple[Path, AdkProgram], monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir, program = generated
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)

    probe.block_network()
    with pytest.raises(OSError, match="blocked"):
        socket.create_connection(("example.com", 80))
    try:
        result = probe.inspect_package(package_dir)
    finally:
        for name in (program.agent_name, f"{program.agent_name}.agent"):
            sys.modules.pop(name, None)

    assert result["root_agent_name"] == program.agent_name
    assert result["app_resumable"] is True
    assert result["exports"] == ["app", "root_agent"]
    assert json.dumps(result)
    shutil.rmtree(package_dir)


def test_probe_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert probe.main([]) == 2
    assert "usage" in capsys.readouterr().err
