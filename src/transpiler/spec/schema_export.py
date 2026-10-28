"""Rules that need code (reserved names, inline secrets, cross-field checks) are not in here."""

import json
import sys
from typing import Any

from transpiler.spec.models import AgentSpec

SCHEMA_ID = "https://lab-exam-agent.local/schemas/agent-spec-1.0.json"


def agent_spec_schema() -> dict[str, Any]:
    schema = AgentSpec.model_json_schema(by_alias=True, mode="validation")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **schema,
    }


def render_schema() -> str:
    return json.dumps(agent_spec_schema(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    sys.stdout.write(render_schema())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
