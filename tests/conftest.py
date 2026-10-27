import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from e2e.helpers import ComposeStack
    from runtime_stack import Stack

# Every test must declare its category so `pytest -m ...` selections in the
# development check and CI never silently skip an unmarked test.
CATEGORY_MARKERS = frozenset({"unit", "contract", "integration", "security", "e2e", "llm"})


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    unmarked = [
        item.nodeid
        for item in items
        if not CATEGORY_MARKERS.intersection(m.name for m in item.iter_markers())
    ]
    if unmarked:
        expected = ", ".join(sorted(CATEGORY_MARKERS))
        listing = "\n  ".join(unmarked)
        raise pytest.UsageError(f"tests without a category marker ({expected}):\n  {listing}")


@pytest.fixture(scope="session")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator["Stack"]:
    """Postgres, scheduler API, OCR and RAG servers plus a freshly generated agent.

    Imported lazily: unit test runs never pay for the servers' imports.
    """
    from runtime_stack import runtime_stack  # noqa: PLC0415

    with runtime_stack(Path(tmp_path_factory.mktemp("runtime"))) as running:
        yield running


@pytest.fixture(scope="session")
def e2e_stack(request: pytest.FixtureRequest) -> Iterator["ComposeStack"]:
    """The Docker Compose stack (project `lab-agent-e2e`), shared by tests/e2e and the
    end-to-end PII checks in tests/security.

    Removed with its volumes at the end, even after a failure; the logs of every
    container are saved to evidence/e2e/ first when a test failed or KEEP_EVIDENCE=1.
    """
    from e2e.helpers import compose_stack, save_logs  # noqa: PLC0415

    with compose_stack() as running:
        try:
            yield running
        finally:
            if request.session.testsfailed or os.environ.get("KEEP_EVIDENCE") == "1":
                save_logs(running)
