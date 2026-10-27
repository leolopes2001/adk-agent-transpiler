from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from runtime_stack import Stack

# Every test must declare its category so `pytest -m ...` selections in the
# Makefile and CI never silently skip an unmarked test.
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
