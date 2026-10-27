.PHONY: setup lint typecheck lint-imports test-unit check golden openapi \
	test test-integration test-security \
	env up down transpile run run-review logs isolation

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
PYTHON_IMAGE = python:3.12.16-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

# Creates .env with random secrets, or appends only the variables it lacks.
# Without a local python3 the script runs in a throwaway container (no network).
env:
	@if command -v python3 >/dev/null 2>&1; then \
		python3 scripts/make_env.py; \
	else \
		docker run --rm --network none --user "$$(id -u):$$(id -g)" \
			-v "$(CURDIR):/work" -w /work $(PYTHON_IMAGE) python scripts/make_env.py; \
	fi

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
