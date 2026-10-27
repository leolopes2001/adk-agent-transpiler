import importlib

import pytest

pytestmark = pytest.mark.unit

PACKAGES = ["lab_common", "transpiler", "agent_runtime", "ocr_mcp", "rag_mcp", "scheduler_api"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name).__name__ == name


@pytest.mark.parametrize("module", ["agent_runtime.cli"])
def test_cli_stub_exits_with_failure(module: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert importlib.import_module(module).main() == 1
    assert "not implemented" in capsys.readouterr().err
