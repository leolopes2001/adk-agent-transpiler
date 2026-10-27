from typing import Any

import pytest
from pydantic import ValidationError

from transpiler.spec.models import (
    AgentSpec,
    ConfidenceGateNode,
    OcrExtractNode,
    RetrySpec,
    find_inline_secrets,
    is_secret_key,
)

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]


def _error_types(exc: pytest.ExceptionInfo[ValidationError]) -> set[str]:
    return {error["type"] for error in exc.value.errors()}


def test_valid_spec_parses(spec_dict: SpecDict) -> None:
    spec = AgentSpec.model_validate(spec_dict)

    assert spec.agent.name == "full_agent"
    assert [node.id for node in spec.workflow.nodes][:2] == ["ocr", "guard"]
    assert spec.workflow.edges[0].source == "START"
    assert set(spec.subworkflows) == {"extraction"}


def test_kind_selects_node_model(spec_dict: SpecDict) -> None:
    spec = AgentSpec.model_validate(spec_dict)

    assert isinstance(spec.workflow.nodes[0], OcrExtractNode)
    assert isinstance(spec.workflow.nodes[4], ConfidenceGateNode)


def test_param_defaults_come_from_the_kind(spec_dict: SpecDict) -> None:
    spec = AgentSpec.model_validate(spec_dict)
    ocr, guard = spec.workflow.nodes[0], spec.workflow.nodes[1]

    assert isinstance(ocr, OcrExtractNode)
    assert ocr.params.min_mean_confidence == 0.5
    assert ocr.params.max_image_bytes == 8 * 1024 * 1024
    assert guard.model_dump()["params"] == {"max_chars": 8000, "block_at": "high"}


def test_policy_defaults(spec_dict: SpecDict) -> None:
    defaults = AgentSpec.model_validate(spec_dict).policies.defaults

    assert defaults.timeout_s == 30
    assert defaults.retry == RetrySpec(max_attempts=1, initial_delay_s=0.5, max_delay_s=2)


def test_models_are_frozen(spec_dict: SpecDict) -> None:
    spec = AgentSpec.model_validate(spec_dict)

    with pytest.raises(ValidationError):
        spec.agent.name = "other"  # type: ignore[misc]


def test_unknown_field_is_rejected(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["color"] = "red"

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"extra_forbidden"}


def test_unknown_kind_is_rejected(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["kind"] = "shell_exec"

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"union_tag_invalid"}


def test_kind_specific_requirement(spec_dict: SpecDict) -> None:
    del spec_dict["workflow"]["nodes"][0]["server"]

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    [error] = exc.value.errors()
    assert error["type"] == "missing"
    assert error["loc"] == ("workflow", "nodes", 0, "ocr_extract", "server")


def test_field_not_allowed_for_kind(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["retry"] = spec_dict["workflow"]["nodes"][0]["retry"]

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"extra_forbidden"}


@pytest.mark.parametrize("name", ["Ocr", "1ocr", "ocr-node", "a" * 64, "", "ocr node"])
def test_invalid_identifier(spec_dict: SpecDict, name: str) -> None:
    spec_dict["workflow"]["nodes"][1]["id"] = name

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"string_pattern_mismatch"}


@pytest.mark.parametrize("name", ["class", "match", "deps", "root_agent", "types", "print", "open"])
def test_reserved_identifier(spec_dict: SpecDict, name: str) -> None:
    spec_dict["workflow"]["nodes"][1]["id"] = name

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"reserved_identifier"}


def test_map_keys_are_identifiers(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["Bad-Key"] = spec_dict["mcp_servers"]["ocr"]

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"string_pattern_mismatch"}


@pytest.mark.parametrize("value", ["ocr_url", "1URL", "X", "URL-1"])
def test_invalid_env_var_name(spec_dict: SpecDict, value: str) -> None:
    spec_dict["mcp_servers"]["ocr"]["url_env"] = value

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"string_pattern_mismatch"}


def test_only_sse_transport(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["ocr"]["transport"] = "streamable_http"

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"literal_error"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("accept_threshold", 1.2), ("review_threshold", -0.1), ("min_margin", 0.6)],
)
def test_gate_ranges(spec_dict: SpecDict, field: str, value: float) -> None:
    spec_dict["workflow"]["nodes"][4]["params"][field] = value

    with pytest.raises(ValidationError):
        AgentSpec.model_validate(spec_dict)


def test_review_threshold_above_accept(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][4]["params"]["review_threshold"] = 0.9

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"threshold_order"}


@pytest.mark.parametrize("value", [0, -1, 301])
def test_timeout_range(spec_dict: SpecDict, value: float) -> None:
    spec_dict["workflow"]["nodes"][0]["timeout_s"] = value

    with pytest.raises(ValidationError):
        AgentSpec.model_validate(spec_dict)


@pytest.mark.parametrize(
    "retry",
    [
        {"max_attempts": 0, "initial_delay_s": 0.5, "max_delay_s": 2},
        {"max_attempts": 6, "initial_delay_s": 0.5, "max_delay_s": 2},
        {"max_attempts": 2, "initial_delay_s": 0.05, "max_delay_s": 2},
        {"max_attempts": 2, "initial_delay_s": 0.5, "max_delay_s": 31},
        {"max_attempts": 2, "initial_delay_s": 5, "max_delay_s": 2},
    ],
)
def test_retry_ranges(spec_dict: SpecDict, retry: dict[str, float]) -> None:
    spec_dict["workflow"]["nodes"][0]["retry"] = retry

    with pytest.raises(ValidationError):
        AgentSpec.model_validate(spec_dict)


def test_strict_types_do_not_coerce_strings(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][0]["timeout_s"] = "45"

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"float_type"}


def test_duplicate_allowed_tools(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["rag"]["allowed_tools"] = ["batch_search", "batch_search"]

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"duplicate_item"}


def test_output_schema_is_an_allowlist(spec_dict: SpecDict) -> None:
    spec_dict["llm_agents"]["extractor_agent"]["output_schema"] = "os.system"

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"literal_error"}


def test_edge_uses_from_and_to(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["edges"][0] = {"source": "START", "target": "ocr"}

    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(spec_dict)
    assert _error_types(exc) == {"missing", "extra_forbidden"}


@pytest.mark.parametrize(
    ("key", "secret"),
    [
        ("token", True),
        ("api_key", True),
        ("scheduler_api_key", True),
        ("db_password", True),
        ("client_secret_value", True),
        ("api_key_env", False),
        ("auth_token_env", False),
        ("max_output_tokens", False),
        ("tokenizer", False),
    ],
)
def test_is_secret_key(key: str, secret: bool) -> None:
    assert is_secret_key(key) is secret


def test_find_inline_secrets_reports_paths(spec_dict: SpecDict) -> None:
    spec_dict["http_services"]["scheduler"]["api_key"] = "abc123"
    spec_dict["workflow"]["nodes"][0]["params"]["token"] = "xyz"  # noqa: S105

    assert find_inline_secrets(spec_dict) == [
        ("http_services", "scheduler", "api_key"),
        ("workflow", "nodes", 0, "params", "token"),
    ]


def test_names_in_named_maps_are_not_secrets(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["token_service"] = spec_dict["mcp_servers"]["rag"]

    assert find_inline_secrets(spec_dict) == []
