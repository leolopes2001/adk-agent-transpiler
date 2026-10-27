import json
import shutil
from pathlib import Path

import pytest

from transpiler.cli import main
from transpiler.spec.schema_export import render_schema

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[3]
SPECS = ROOT / "specs"
INVALID = ROOT / "tests" / "data" / "specs" / "invalid"

type Capture = pytest.CaptureFixture[str]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project-like directory with the main spec; the CWD is the allowed output root."""
    shutil.copytree(SPECS, tmp_path / "specs")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_validate_main_spec(capsys: Capture) -> None:
    assert main(["validate", str(SPECS / "exam_scheduling_agent.json")]) == 0
    assert capsys.readouterr().out.startswith("✔ spec válida:")


def test_validate_reports_semantic_error(capsys: Capture) -> None:
    code = main(["validate", str(INVALID / "E_ROUTE_NOT_COVERED.json")])

    err = capsys.readouterr().err.splitlines()
    assert code == 3
    assert err[0] == '✖ E_ROUTE_NOT_COVERED  /workflow/nodes/4  (node "gate", kind confidence_gate)'
    assert err[-1] == "1 erro. Nenhum arquivo foi gerado."


def test_validate_json(capsys: Capture) -> None:
    code = main(["validate", str(INVALID / "E_ROUTE_NOT_COVERED.json"), "--format", "json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 3
    assert document["ok"] is False
    assert document["diagnostics"][0]["code"] == "E_ROUTE_NOT_COVERED"


def test_validate_json_success_with_warning(capsys: Capture) -> None:
    code = main(["validate", str(INVALID / "W_UNUSED_DECLARATION.json"), "--format", "json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (document["ok"], document["warnings"]) == (True, 1)


def test_warnings_do_not_fail(capsys: Capture) -> None:
    assert main(["validate", str(INVALID / "W_UNUSED_DECLARATION.json")]) == 0

    captured = capsys.readouterr()
    assert "W_UNUSED_DECLARATION" in captured.err
    assert "(1 aviso(s))" in captured.out


def test_inline_secret_exit_code(tmp_path: Path, capsys: Capture) -> None:
    spec = json.loads((SPECS / "exam_scheduling_agent.json").read_text(encoding="utf-8"))
    spec["http_services"]["scheduler"]["api_key"] = "hunter2"
    path = tmp_path / "secret.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    assert main(["validate", str(path)]) == 2
    err = capsys.readouterr().err
    assert "E_INLINE_SECRET" in err
    assert "hunter2" not in err


def test_schema_error_exit_code(tmp_path: Path, capsys: Capture) -> None:
    spec = tmp_path / "bad.json"
    spec.write_text('{"spec_version": "9"}', encoding="utf-8")

    assert main(["validate", str(spec)]) == 2


def test_missing_file_is_io_error(tmp_path: Path, capsys: Capture) -> None:
    assert main(["validate", str(tmp_path / "nope.json")]) == 1
    assert "E_IO" in capsys.readouterr().err


def test_usage_error_exits_1(capsys: Capture) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 1


def test_schema_command(capsys: Capture) -> None:
    assert main(["schema"]) == 0
    assert capsys.readouterr().out == render_schema()


def test_generate(workdir: Path, capsys: Capture) -> None:
    code = main(["generate", "specs/exam_scheduling_agent.json", "--out", "generated/"])

    package = workdir / "generated" / "exam_scheduling_agent"
    assert code == 0
    assert sorted(p.name for p in package.iterdir()) == [
        "__init__.py",
        "_manifest.json",
        "agent.py",
    ]
    manifest = json.loads((package / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec_path"] == "specs/exam_scheduling_agent.json"
    assert manifest["adk_requirement"].startswith("google-adk==")
    assert set(manifest["instructions"]) == {"exam_extractor"}
    assert "extraction" in manifest["node_names"]
    out = capsys.readouterr().out
    assert out == (
        "✔ gerado: generated/exam_scheduling_agent (agent.py, __init__.py, _manifest.json)\n"
    )


def test_generate_json(workdir: Path, capsys: Capture) -> None:
    code = main(["generate", "specs/exam_scheduling_agent.json", "--format", "json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["files"] == ["agent.py", "__init__.py", "_manifest.json"]
    assert document["output"].endswith("generated/exam_scheduling_agent")


def test_generate_invalid_writes_nothing(workdir: Path, capsys: Capture) -> None:
    shutil.copy(INVALID / "E_ROUTE_NOT_COVERED.json", workdir / "specs" / "bad.json")

    assert main(["generate", "specs/bad.json", "--out", "generated"]) == 3
    assert not (workdir / "generated").exists()
    assert "Nenhum arquivo foi gerado." in capsys.readouterr().err


def test_generate_out_must_stay_in_project(workdir: Path, capsys: Capture) -> None:
    code = main(["generate", "specs/exam_scheduling_agent.json", "--out", "../elsewhere"])

    assert code == 1
    assert "E_OUTPUT" in capsys.readouterr().err
    assert not (workdir.parent / "elsewhere").exists()
