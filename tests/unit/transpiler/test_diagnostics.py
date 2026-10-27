import json
from typing import Any

import pytest

from transpiler.errors import (
    Code,
    Diagnostic,
    Severity,
    Stage,
    TranspileError,
    fail_on_errors,
    json_pointer,
    render_json,
    render_text,
    summary,
)
from transpiler.spec.loader import parse_spec

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]


def _diagnostics(raw: SpecDict) -> tuple[Diagnostic, ...]:
    with pytest.raises(TranspileError) as exc:
        parse_spec(raw)
    assert exc.value.stage is Stage.SCHEMA
    return exc.value.diagnostics


def _single(raw: SpecDict) -> Diagnostic:
    [diagnostic] = _diagnostics(raw)
    return diagnostic


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ((), ""),
        (("workflow", "nodes", 3, "params", "top_k"), "/workflow/nodes/3/params/top_k"),
        (("a/b", "c~d"), "/a~1b/c~0d"),
    ],
)
def test_json_pointer(path: tuple[str | int, ...], expected: str) -> None:
    assert json_pointer(path) == expected


def test_missing_kind_field_points_at_node(spec_dict: SpecDict) -> None:
    del spec_dict["workflow"]["nodes"][0]["server"]

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_MISSING_FIELD
    assert diagnostic.pointer == "/workflow/nodes/0/server"
    assert diagnostic.message == 'ocr_extract requer "server"'
    assert (diagnostic.node_id, diagnostic.kind) == ("ocr", "ocr_extract")


def test_param_out_of_range(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][3]["params"] = {"top_k": 50}

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_SCHEMA
    assert diagnostic.pointer == "/workflow/nodes/3/params/top_k"
    assert diagnostic.node_id == "resolve"


@pytest.mark.parametrize("top_k", [11, 20])
def test_top_k_above_the_rag_limit_is_rejected(spec_dict: SpecDict, top_k: int) -> None:
    spec_dict["workflow"]["nodes"][3]["params"] = {"top_k": top_k}

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_SCHEMA
    assert diagnostic.pointer == "/workflow/nodes/3/params/top_k"
    assert "10" in diagnostic.message


def test_top_k_at_the_rag_limit_is_accepted(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][3]["params"] = {"top_k": 10}

    parse_spec(spec_dict)


def test_unknown_field_in_subworkflow(spec_dict: SpecDict) -> None:
    spec_dict["subworkflows"]["extraction"]["nodes"][2]["retry"] = {}

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_UNKNOWN_FIELD
    assert diagnostic.pointer == "/subworkflows/extraction/nodes/2/retry"
    assert diagnostic.message == 'campo "retry" não é permitido em nós join'


def test_unknown_kind(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["kind"] = "shell"

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_UNKNOWN_KIND
    assert diagnostic.pointer == "/workflow/nodes/1"
    assert diagnostic.hint is not None
    assert "ocr_extract" in diagnostic.hint


def test_missing_kind(spec_dict: SpecDict) -> None:
    del spec_dict["workflow"]["nodes"][1]["kind"]

    assert _single(spec_dict).code is Code.E_MISSING_FIELD


def test_invalid_identifier(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["id"] = "Guard"

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_INVALID_IDENTIFIER
    assert diagnostic.pointer == "/workflow/nodes/1/id"


def test_invalid_map_key_points_at_entry(spec_dict: SpecDict) -> None:
    spec_dict["models"]["Bad"] = spec_dict["models"]["extractor"]

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_INVALID_IDENTIFIER
    assert diagnostic.pointer == "/models/Bad"


def test_reserved_identifier(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][1]["id"] = "deps"

    assert _single(spec_dict).code is Code.E_RESERVED_IDENTIFIER


def test_invalid_env_name(spec_dict: SpecDict) -> None:
    spec_dict["http_services"]["scheduler"]["api_key_env"] = "key"

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_INVALID_ENV_NAME
    assert diagnostic.pointer == "/http_services/scheduler/api_key_env"


def test_threshold_order(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][4]["params"]["review_threshold"] = 0.95

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_THRESHOLD_ORDER
    assert diagnostic.pointer == "/workflow/nodes/4/params"


def test_inline_secret_replaces_unknown_field(spec_dict: SpecDict) -> None:
    spec_dict["http_services"]["scheduler"]["api_key"] = "s3cr3t"

    diagnostic = _single(spec_dict)

    assert diagnostic.code is Code.E_INLINE_SECRET
    assert diagnostic.pointer == "/http_services/scheduler/api_key"
    assert "s3cr3t" not in render_text([diagnostic])


def test_all_errors_are_reported(spec_dict: SpecDict) -> None:
    spec_dict["mcp_servers"]["ocr"]["token"] = "abc"  # noqa: S105
    spec_dict["workflow"]["nodes"][1]["id"] = "Guard"
    spec_dict["workflow"]["nodes"][4]["params"]["min_margin"] = 0.9
    del spec_dict["subworkflows"]["extraction"]["nodes"][0]["agent"]

    codes = {(d.code, d.pointer) for d in _diagnostics(spec_dict)}

    assert codes == {
        (Code.E_INLINE_SECRET, "/mcp_servers/ocr/token"),
        (Code.E_INVALID_IDENTIFIER, "/workflow/nodes/1/id"),
        (Code.E_SCHEMA, "/workflow/nodes/4/params/min_margin"),
        (Code.E_MISSING_FIELD, "/subworkflows/extraction/nodes/0/agent"),
    }


def test_severity_from_prefix() -> None:
    assert Diagnostic(Code.E_DEAD_END, "x").severity is Severity.ERROR
    assert Diagnostic(Code.W_UNUSED_DECLARATION, "x").severity is Severity.WARNING


def test_codes_are_unique_and_prefixed() -> None:
    assert all(code.value == code.name for code in Code)
    assert all(code.startswith(("E_", "W_")) for code in Code)


def test_fail_on_errors_ignores_warnings_only() -> None:
    warning = Diagnostic(Code.W_UNUSED_DECLARATION, "x")
    error = Diagnostic(Code.E_DEAD_END, "y")

    fail_on_errors(Stage.SEMANTIC, [warning])
    with pytest.raises(TranspileError) as exc:
        fail_on_errors(Stage.SEMANTIC, [warning, error])
    assert exc.value.diagnostics == (warning, error)
    assert exc.value.exit_code == 3


DIAGNOSTICS = [
    Diagnostic(
        Code.E_ROUTE_NOT_COVERED,
        'A rota "empty" emitida por confidence_gate não possui aresta de saída.',
        pointer="/workflow/nodes/4",
        hint='adicione {"from": "gate", "to": "<nó>", "route": "empty"}',
        node_id="gate",
        kind="confidence_gate",
    ),
    Diagnostic(
        Code.E_TOOL_NOT_ALLOWED, "A tool não é permitida.", pointer="/workflow/nodes/3/tool"
    ),
    Diagnostic(Code.W_UNUSED_DECLARATION, "modelo não usado", pointer="/models/x"),
]


def test_render_text() -> None:
    assert render_text(DIAGNOSTICS).splitlines() == [
        '✖ E_ROUTE_NOT_COVERED   /workflow/nodes/4  (node "gate", kind confidence_gate)',
        '  A rota "empty" emitida por confidence_gate não possui aresta de saída.',
        '  dica: adicione {"from": "gate", "to": "<nó>", "route": "empty"}',
        "✖ E_TOOL_NOT_ALLOWED    /workflow/nodes/3/tool",
        "  A tool não é permitida.",
        "⚠ W_UNUSED_DECLARATION  /models/x",
        "  modelo não usado",
    ]


def test_summary() -> None:
    assert summary(DIAGNOSTICS) == "2 erros, 1 aviso."
    assert summary(DIAGNOSTICS[:1]) == "1 erro."


def test_render_json() -> None:
    document = json.loads(render_json(DIAGNOSTICS))

    assert document["ok"] is False
    assert (document["errors"], document["warnings"]) == (2, 1)
    assert document["diagnostics"][0] == {
        "code": "E_ROUTE_NOT_COVERED",
        "severity": "error",
        "pointer": "/workflow/nodes/4",
        "message": 'A rota "empty" emitida por confidence_gate não possui aresta de saída.',
        "hint": 'adicione {"from": "gate", "to": "<nó>", "route": "empty"}',
        "node_id": "gate",
        "kind": "confidence_gate",
    }
