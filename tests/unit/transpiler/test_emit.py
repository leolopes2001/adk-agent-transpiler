import json
import os
from pathlib import Path

import pytest

from transpiler.emit import build_manifest, resolve_output_root, staged_package
from transpiler.errors import Code, Stage, TranspileError

pytestmark = pytest.mark.unit

MANIFEST_ARGS = {
    "transpiler_version": "0.1.0",
    "spec_path": "agent_specs/a.json",
    "spec_sha256": "a" * 64,
    "description": "Agente de exemplo.",
    "ir_digest": "b" * 64,
    "instructions": {"z_agent": "c" * 64, "a_agent": "d" * 64},
    "adk_requirement": "google-adk==2.13.0",
    "node_names": ["ocr", "abort", "gate"],
}


def _leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.name.startswith(".tmp-"))


def test_commit_writes_package(tmp_path: Path) -> None:
    with staged_package(tmp_path / "generated", "agent_x") as stage:
        stage.write("agent.py", "x = 1\n")
        stage.write("__init__.py", "")
        target = stage.commit()

    assert target == tmp_path / "generated" / "agent_x"
    assert (target / "agent.py").read_text(encoding="utf-8") == "x = 1\n"
    assert _leftovers(tmp_path / "generated") == []


def test_commit_replaces_previous_package(tmp_path: Path) -> None:
    old = tmp_path / "agent_x"
    old.mkdir()
    (old / "stale.py").write_text("old", encoding="utf-8")

    with staged_package(tmp_path, "agent_x") as stage:
        stage.write("agent.py", "new\n")
        stage.commit()

    assert sorted(p.name for p in old.iterdir()) == ["agent.py"]
    assert _leftovers(tmp_path) == []


def test_failure_leaves_previous_package_untouched(tmp_path: Path) -> None:
    old = tmp_path / "agent_x"
    old.mkdir()
    (old / "agent.py").write_text("old\n", encoding="utf-8")

    def generate() -> None:
        with staged_package(tmp_path, "agent_x") as stage:
            stage.write("agent.py", "half")
            raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError):
        generate()

    assert (old / "agent.py").read_text(encoding="utf-8") == "old\n"
    assert _leftovers(tmp_path) == []


def test_without_commit_nothing_is_written(tmp_path: Path) -> None:
    with staged_package(tmp_path, "agent_x") as stage:
        stage.write("agent.py", "x\n")

    assert not (tmp_path / "agent_x").exists()
    assert _leftovers(tmp_path) == []


def test_staged_files_are_importable_by_name(tmp_path: Path) -> None:
    # The probe imports the package by its directory name, so staging keeps it.
    with staged_package(tmp_path, "agent_x") as stage:
        assert stage.path.name == "agent_x"
        assert stage.path.parent.name.startswith(".tmp-")


def test_files_must_be_flat(tmp_path: Path) -> None:
    with staged_package(tmp_path, "agent_x") as stage, pytest.raises(ValueError, match="flat"):
        stage.write("../escape.py", "x")


def test_symlink_target_is_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, tmp_path / "agent_x")

    with staged_package(tmp_path, "agent_x") as stage:
        stage.write("agent.py", "x\n")
        with pytest.raises(TranspileError) as exc:
            stage.commit()
    assert exc.value.diagnostics[0].code is Code.E_OUTPUT
    assert list(elsewhere.iterdir()) == []


def test_output_root_must_be_inside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "project"
    allowed.mkdir()

    assert resolve_output_root(allowed / "generated", allowed) == (allowed / "generated").resolve()
    with pytest.raises(TranspileError) as exc:
        resolve_output_root(allowed / ".." / "outside", allowed)
    assert exc.value.stage is Stage.IO
    assert exc.value.diagnostics[0].code is Code.E_OUTPUT


def test_output_root_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "project"
    allowed.mkdir()
    os.symlink(tmp_path, allowed / "link")

    with pytest.raises(TranspileError):
        resolve_output_root(allowed / "link" / "generated", allowed)


def test_output_root_must_be_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")

    with pytest.raises(TranspileError):
        resolve_output_root(tmp_path / "file", tmp_path)


def test_manifest_is_canonical() -> None:
    text = build_manifest(**MANIFEST_ARGS)  # type: ignore[arg-type]
    manifest = json.loads(text)

    assert text.endswith("}\n")
    assert list(manifest) == [
        "transpiler_version",
        "spec_path",
        "spec_sha256",
        "description",
        "ir_digest",
        "instructions",
        "adk_requirement",
        "node_names",
    ]
    assert list(manifest["instructions"]) == ["a_agent", "z_agent"]
    assert manifest["node_names"] == ["abort", "gate", "ocr"]
    assert build_manifest(**MANIFEST_ARGS) == text  # type: ignore[arg-type]
