.PHONY: setup lint typecheck lint-imports test-unit check golden openapi \
	test test-integration test-security \
	up down transpile run run-review logs isolation

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

# Integration and security tests start Postgres with testcontainers (needs Docker).
test-integration:
	uv run pytest -m integration

test-security:
	uv run pytest -m security

test:
	uv run pytest -m "unit or contract or integration or security"

check: lint typecheck lint-imports test-unit

# Regenerates tests/golden from the versioned specs; review the diff before committing.
golden:
	uv run lab-transpile generate specs/exam_scheduling_agent.json --out tests/golden/
	uv run lab-transpile generate specs/examples/exam_lookup_only.json --out tests/golden/

# Regenerates the versioned OpenAPI document; test_openapi fails when it is stale.
openapi:
	uv run python -m scheduler_api.openapi evidence/openapi.json

# --- Docker Compose (needs only Docker; the raw commands are in the README) ---

COMPOSE ?= docker compose
IMAGE ?= pedido_medico_ficticio.png
# Builds the agent image too: `up` skips it because it is a CLI (profile cli).
up:
	$(COMPOSE) --profile cli build
	$(COMPOSE) up -d --wait

# Also removes the database volume.
down:
	$(COMPOSE) --profile cli down -v

transpile:
	$(COMPOSE) run --rm --entrypoint lab-transpile agent \
		generate /app/specs/exam_scheduling_agent.json --out /app/generated

run:
	$(COMPOSE) run --rm agent run --image /app/inputs/samples/$(IMAGE)

# Interactive: the ambiguous order asks which exam was meant.
run-review:
	$(COMPOSE) run --rm agent run --image /app/inputs/samples/pedido_ambiguo.png

logs:
	$(COMPOSE) logs -f --tail=100

isolation:
	COMPOSE="$(COMPOSE)" bash scripts/check_isolation.sh
