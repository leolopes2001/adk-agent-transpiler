import pytest

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
