import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from transpiler.spec.schema_export import agent_spec_schema, render_schema

pytestmark = pytest.mark.unit

SCHEMA_FILE = Path(__file__).parents[3] / "schemas" / "agent-spec.schema.json"
REGENERATE = "uv run python -m transpiler.spec.schema_export > schemas/agent-spec.schema.json"


def test_versioned_schema_matches_models() -> None:
    assert SCHEMA_FILE.read_text(encoding="utf-8") == render_schema(), (
        f"schema drift; regenerate with: {REGENERATE}"
    )


def test_schema_is_valid_draft_2020_12() -> None:
    jsonschema.Draft202012Validator.check_schema(agent_spec_schema())


def test_valid_spec_passes_schema(spec_dict: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator(agent_spec_schema()).validate(spec_dict)


def test_schema_rejects_extra_fields(spec_dict: dict[str, Any]) -> None:
    spec_dict["workflow"]["nodes"][0]["shell"] = "rm -rf /"
    validator = jsonschema.Draft202012Validator(agent_spec_schema())

    assert not validator.is_valid(spec_dict)


def test_schema_uses_json_field_names() -> None:
    edge = agent_spec_schema()["$defs"]["EdgeSpec"]

    assert set(edge["properties"]) == {"from", "to", "route"}
    assert json.dumps(edge).count("source") == 0
