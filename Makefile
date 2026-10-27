.PHONY: setup lint typecheck lint-imports test-unit check golden

setup:
	uv sync --all-extras --group dev
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

lint-imports:
	uv run lint-imports

test-unit:
	uv run pytest -m "unit or contract"

check: lint typecheck lint-imports test-unit

# Regenerates tests/golden from the versioned specs; review the diff before committing.
golden:
	uv run lab-transpile generate specs/exam_scheduling_agent.json --out tests/golden/
	uv run lab-transpile generate specs/examples/exam_lookup_only.json --out tests/golden/
