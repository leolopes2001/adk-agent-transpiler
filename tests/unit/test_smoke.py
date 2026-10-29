import importlib

import pytest

pytestmark = pytest.mark.unit

PACKAGES = ["lab_common", "transpiler", "agent_runtime", "ocr_mcp", "rag_mcp", "scheduler_api"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name).__name__ == name
