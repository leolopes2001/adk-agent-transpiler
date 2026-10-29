from collections.abc import Callable
from typing import Any

import pytest

from transpiler.errors import Code, Diagnostic

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]
type Check = Callable[[SpecDict], list[Diagnostic]]


def _found(diagnostics: list[Diagnostic]) -> set[tuple[str, str]]:
    return {(d.code.value, d.pointer) for d in diagnostics}


def _codes(diagnostics: list[Diagnostic], code: Code) -> list[Diagnostic]:
    return [d for d in diagnostics if d.code is code]


CORPUS = {
    "E_UNKNOWN_ROUTE": "/workflow/edges/2/route",
    "E_ROUTE_NOT_COVERED": "/workflow/nodes/4",
    "E_ROUTE_ON_NON_ROUTER": "/workflow/edges/6/route",
    "E_MIXED_ROUTED_EDGES": "/workflow/edges/13",
}


@pytest.mark.parametrize(("code", "pointer"), CORPUS.items())
def test_invalid_corpus(
    check_corpus: Callable[[str], list[Diagnostic]], code: str, pointer: str
) -> None:
    assert (code, pointer) in _found(check_corpus(code))


def test_unknown_route_lists_valid_routes(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"][7]["route"] = "approved"

    [unknown] = _codes(check(spec_dict), Code.E_UNKNOWN_ROUTE)
    assert unknown.hint == "rotas de confidence_gate: accepted, review, empty"
    assert unknown.node_id == "gate"


def test_route_not_covered_message(spec_dict: SpecDict, check: Check) -> None:
    del spec_dict["workflow"]["edges"][9]

    [diagnostic] = check(spec_dict)
    assert diagnostic.code is Code.E_ROUTE_NOT_COVERED
    assert diagnostic.message == (
        'A rota "empty" emitida por confidence_gate não possui aresta de saída.'
    )
    assert diagnostic.hint == 'adicione {"from": "gate", "to": "<nó>", "route": "empty"}'


def test_each_missing_route_is_reported(spec_dict: SpecDict, check: Check) -> None:
    edges = spec_dict["workflow"]["edges"]
    spec_dict["workflow"]["edges"] = [e for e in edges if e["from"] != "review"]

    missing = _codes(check(spec_dict), Code.E_ROUTE_NOT_COVERED)
    assert [d.message.split('"')[1] for d in missing] == ["confirmed", "rejected"]


def test_route_from_start(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"][0]["route"] = "go"

    assert _found(check(spec_dict)) == {("E_ROUTE_ON_NON_ROUTER", "/workflow/edges/0/route")}


def test_route_on_subworkflow_node(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"][5]["route"] = "ok"

    assert ("E_ROUTE_ON_NON_ROUTER", "/workflow/edges/5/route") in _found(check(spec_dict))


def test_routes_in_subworkflow_are_checked(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["subworkflows"]["extraction"]["edges"][4]["route"] = "done"

    assert _found(check(spec_dict)) == {
        ("E_ROUTE_ON_NON_ROUTER", "/subworkflows/extraction/edges/4/route")
    }
