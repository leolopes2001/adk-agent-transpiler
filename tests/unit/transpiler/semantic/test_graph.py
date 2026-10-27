from collections.abc import Callable
from typing import Any

import pytest

from transpiler.errors import Code, Diagnostic

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]
type Check = Callable[[SpecDict], list[Diagnostic]]

ROOT_NODES = ["ocr", "guard", "extraction", "resolve", "gate", "review", "schedule", "present"]


def _found(diagnostics: list[Diagnostic]) -> set[tuple[str, str]]:
    return {(d.code.value, d.pointer) for d in diagnostics}


def _edges(spec: SpecDict, graph: str = "workflow") -> list[dict[str, str]]:
    edges: list[dict[str, str]] = (
        spec["workflow"]["edges"] if graph == "workflow" else spec["subworkflows"][graph]["edges"]
    )
    return edges


CORPUS = {
    "E_DUPLICATE_ID": "/workflow/nodes/9/id",
    "E_UNKNOWN_NODE": "/workflow/edges/12/to",
    "E_NO_START": "/workflow/edges",
    "E_START_TARGET_INVALID": "/workflow/edges/13/to",
    "E_UNREACHABLE": "/workflow/nodes/9",
    "E_DEAD_END": "/workflow/nodes/6",
    "E_TERMINAL_HAS_OUTGOING": "/workflow/nodes/7",
    "E_DUPLICATE_EDGE": "/workflow/edges/13/route",
    "E_FAN_IN_REQUIRES_JOIN": "/subworkflows/extraction/nodes/2",
    "E_JOIN_ARITY": "/workflow/nodes/9",
    "E_UNCONDITIONAL_CYCLE": "/workflow/nodes/2",
    "E_SUBWORKFLOW_CYCLE": "/subworkflows/extraction/nodes/4/ref",
}


@pytest.mark.parametrize(("code", "pointer"), CORPUS.items())
def test_invalid_corpus(
    check_corpus: Callable[[str], list[Diagnostic]], code: str, pointer: str
) -> None:
    assert (code, pointer) in _found(check_corpus(code))


def test_valid_spec_has_no_graph_diagnostics(spec_dict: SpecDict, check: Check) -> None:
    assert check(spec_dict) == []


def test_duplicate_id_in_graph(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"].append({"id": "guard", "kind": "abort"})

    [diagnostic] = check(spec_dict)
    assert diagnostic.code is Code.E_DUPLICATE_ID
    assert diagnostic.pointer == "/workflow/nodes/9/id"


def test_same_id_in_two_graphs_is_a_duplicate_adk_name(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["subworkflows"]["extraction"]["nodes"][3]["id"] = "resolve"
    _edges(spec_dict, "extraction")[4]["to"] = "resolve"

    # Names are claimed in emission order, and the subworkflow is expanded where it is used.
    assert _found(check(spec_dict)) == {("E_DUPLICATE_ID", "/workflow/nodes/3/id")}


def test_lowering_names_are_reserved(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][1]["id"] = "llm_extract_input"
    for edge in _edges(spec_dict):
        for end in ("from", "to"):
            if edge[end] == "guard":
                edge[end] = "llm_extract_input"

    [diagnostic] = check(spec_dict)
    assert diagnostic.code is Code.E_DUPLICATE_ID
    assert diagnostic.pointer == "/subworkflows/extraction/nodes/0/id"
    assert "llm_extract_input" in diagnostic.message


def test_agent_name_collides_with_node(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["llm_agents"]["present"] = spec_dict["llm_agents"].pop("extractor_agent")
    spec_dict["subworkflows"]["extraction"]["nodes"][0]["agent"] = "present"

    assert _found(check(spec_dict)) == {("E_DUPLICATE_ID", "/workflow/nodes/7/id")}


def test_root_name_collides_with_node(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["agent"]["name"] = "gate"

    assert _found(check(spec_dict)) == {("E_DUPLICATE_ID", "/workflow/nodes/4/id")}


def test_subworkflow_referenced_twice_duplicates_inner_names(
    spec_dict: SpecDict, check: Check
) -> None:
    spec_dict["workflow"]["nodes"].append(
        {"id": "again", "kind": "subworkflow", "ref": "extraction"}
    )
    _edges(spec_dict)[5] = {"from": "extraction", "to": "again"}
    _edges(spec_dict).append({"from": "again", "to": "resolve"})

    pointers = {d.pointer for d in check(spec_dict) if d.code is Code.E_DUPLICATE_ID}
    assert "/subworkflows/extraction/nodes/3/id" in pointers


def test_unknown_source_and_target(spec_dict: SpecDict, check: Check) -> None:
    _edges(spec_dict).append({"from": "ghost", "to": "phantom"})

    assert _found(check(spec_dict)) == {
        ("E_UNKNOWN_NODE", "/workflow/edges/13/from"),
        ("E_UNKNOWN_NODE", "/workflow/edges/13/to"),
    }


def test_no_start_in_subworkflow(spec_dict: SpecDict, check: Check) -> None:
    edges = _edges(spec_dict, "extraction")
    edges[0]["from"] = "lexical_extract"
    del edges[1]

    found = _found(check(spec_dict))
    assert ("E_NO_START", "/subworkflows/extraction/edges") in found


def test_unreachable_cluster(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"].append({"id": "late", "kind": "present"})
    spec_dict["workflow"]["nodes"].append({"id": "later", "kind": "untrusted_text_guard"})
    _edges(spec_dict).append({"from": "later", "to": "late", "route": "clean"})
    _edges(spec_dict).append({"from": "later", "to": "abort", "route": "blocked"})

    assert _found(check(spec_dict)) == {
        ("E_UNREACHABLE", "/workflow/nodes/9"),
        ("E_UNREACHABLE", "/workflow/nodes/10"),
    }


def test_subworkflow_needs_single_exit(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["subworkflows"]["extraction"]["nodes"].append({"id": "extra", "kind": "reconcile"})
    _edges(spec_dict, "extraction").append({"from": "merge", "to": "extra"})

    found = _found(check(spec_dict))
    assert ("E_DEAD_END", "/subworkflows/extraction/nodes/3") in found
    assert ("E_DEAD_END", "/subworkflows/extraction/nodes/4") in found


def test_repeated_unconditional_edge(spec_dict: SpecDict, check: Check) -> None:
    _edges(spec_dict).append({"from": "schedule", "to": "present"})

    found = _found(check(spec_dict))
    assert ("E_DUPLICATE_EDGE", "/workflow/edges/13") in found


def test_routed_fan_in_is_allowed(spec_dict: SpecDict, check: Check) -> None:
    # abort and schedule already receive several routed edges in the fixture.
    assert [d for d in check(spec_dict) if d.code is Code.E_FAN_IN_REQUIRES_JOIN] == []


def test_join_counts_distinct_predecessors(spec_dict: SpecDict, check: Check) -> None:
    _edges(spec_dict, "extraction")[3]["from"] = "llm_extract"
    _edges(spec_dict, "extraction").append({"from": "lexical_extract", "to": "reconcile"})

    found = _found(check(spec_dict))
    assert ("E_JOIN_ARITY", "/subworkflows/extraction/nodes/2") in found


def test_self_loop_is_a_cycle(spec_dict: SpecDict, check: Check) -> None:
    _edges(spec_dict).append({"from": "resolve", "to": "resolve"})

    found = _found(check(spec_dict))
    assert ("E_UNCONDITIONAL_CYCLE", "/workflow/nodes/3") in found


def test_cycle_through_a_route_is_allowed(spec_dict: SpecDict, check: Check) -> None:
    # review → (confirmed) schedule is routed, so going back from schedule to review
    # is a conditional cycle.
    spec_dict["workflow"]["nodes"][6] = {"id": "schedule", "kind": "human_review"}
    _edges(spec_dict)[12] = {"from": "schedule", "to": "review", "route": "confirmed"}
    _edges(spec_dict).append({"from": "schedule", "to": "present", "route": "rejected"})

    assert [d for d in check(spec_dict) if d.code is Code.E_UNCONDITIONAL_CYCLE] == []


def test_indirect_subworkflow_cycle(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["subworkflows"]["inner"] = {
        "nodes": [{"id": "back", "kind": "subworkflow", "ref": "extraction"}],
        "edges": [{"from": "START", "to": "back"}],
    }
    spec_dict["subworkflows"]["extraction"]["nodes"].append(
        {"id": "down", "kind": "subworkflow", "ref": "inner"}
    )
    _edges(spec_dict, "extraction").append({"from": "reconcile", "to": "down"})

    found = _found(check(spec_dict))
    assert ("E_SUBWORKFLOW_CYCLE", "/subworkflows/extraction/nodes/4/ref") in found
    assert ("E_SUBWORKFLOW_CYCLE", "/subworkflows/inner/nodes/0/ref") in found


def test_all_graph_errors_are_reported(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"].append({"id": "orphan", "kind": "abort"})
    _edges(spec_dict).append({"from": "present", "to": "ghost"})
    del _edges(spec_dict)[12]

    codes = {d.code for d in check(spec_dict)}
    assert {
        Code.E_UNREACHABLE,
        Code.E_UNKNOWN_NODE,
        Code.E_TERMINAL_HAS_OUTGOING,
        Code.E_DEAD_END,
    } <= codes


def test_diagnostics_name_the_node(spec_dict: SpecDict, check: Check) -> None:
    del _edges(spec_dict)[12]

    [diagnostic] = check(spec_dict)
    assert (diagnostic.node_id, diagnostic.kind) == ("schedule", "schedule")
    assert ROOT_NODES.index("schedule") == 6
