import dataclasses
from typing import Any

import pytest

from transpiler.ir.builder import build_program
from transpiler.ir.model import RetryIR
from transpiler.lowering import AdkEdge, AdkGraph, AdkNode, AdkProgram, lower
from transpiler.spec.loader import InstructionFile, parse_spec

pytestmark = pytest.mark.unit

type SpecDict = dict[str, Any]

INSTRUCTION = InstructionFile("instructions/extractor.md", "Extraia exames.\n", "ab" * 32)


def _lower(spec: SpecDict) -> AdkProgram:
    program = build_program(
        parse_spec(spec),
        spec_sha256="0" * 64,
        instructions={name: INSTRUCTION for name in spec["llm_agents"]},
    )
    return lower(program)


@pytest.fixture
def adk(spec_dict: SpecDict) -> AdkProgram:
    return _lower(spec_dict)


def _node(graph: AdkGraph, var: str) -> AdkNode:
    return next(node for node in graph.nodes if node.var == var)


def _extraction(adk: AdkProgram) -> AdkGraph:
    subgraph = _node(adk.root, "extraction").subgraph
    assert subgraph is not None
    return subgraph


def test_llm_extract_becomes_input_node_and_agent(adk: AdkProgram) -> None:
    extraction = _extraction(adk)

    assert [(n.var, n.primitive) for n in extraction.nodes] == [
        ("llm_extract_input", "function"),
        ("extractor_agent", "agent"),
        ("lexical_extract", "function"),
        ("merge", "join"),
        ("reconcile", "function"),
    ]
    assert _node(extraction, "llm_extract_input").factory == "llm_input"
    agent = _node(extraction, "extractor_agent").agent
    assert agent is not None
    assert agent.key == "extractor_agent"


def test_llm_extract_edges_are_rewired(adk: AdkProgram) -> None:
    assert _extraction(adk).edges == (
        AdkEdge("START", "llm_extract_input"),
        AdkEdge("llm_extract_input", "extractor_agent"),
        AdkEdge("START", "lexical_extract"),
        AdkEdge("extractor_agent", "merge"),
        AdkEdge("lexical_extract", "merge"),
        AdkEdge("merge", "reconcile"),
    )


def test_reconcile_sources_name_the_join_predecessors(adk: AdkProgram) -> None:
    reconcile = _node(_extraction(adk), "reconcile")

    assert dict(reconcile.kwargs) == {
        "sources": {"llm": "extractor_agent", "lexical": "lexical_extract"},
        "grounding_min_ratio": 90,
        "max_items": 30,
    }


def test_reconcile_sources_follow_spec_names(spec_dict: SpecDict) -> None:
    sub = spec_dict["subworkflows"]["extraction"]
    sub["nodes"][1]["id"] = "words"
    sub["edges"][1]["to"] = "words"
    sub["edges"][3]["from"] = "words"
    # Join inputs listed lexical first: the roles still come from the predecessor types.
    sub["edges"][2], sub["edges"][3] = sub["edges"][3], sub["edges"][2]

    reconcile = _node(_extraction(_lower(spec_dict)), "reconcile")
    assert reconcile.kwargs["sources"] == {"llm": "extractor_agent", "lexical": "words"}


def test_routers_become_route_dicts(adk: AdkProgram) -> None:
    assert adk.root.edges == (
        AdkEdge("START", "ocr"),
        AdkEdge("ocr", routes=(("ok", "guard"), ("low_quality", "abort"))),
        AdkEdge("guard", routes=(("clean", "extraction"), ("blocked", "abort"))),
        AdkEdge("extraction", "resolve"),
        AdkEdge("resolve", "gate"),
        AdkEdge(
            "gate", routes=(("accepted", "schedule"), ("review", "review"), ("empty", "abort"))
        ),
        AdkEdge("review", routes=(("confirmed", "schedule"), ("rejected", "present"))),
        AdkEdge("schedule", "present"),
    )


def test_route_dict_sits_at_first_route_edge(spec_dict: SpecDict) -> None:
    edges = spec_dict["workflow"]["edges"]
    edges.append(edges.pop(2))  # ocr's low_quality edge moves to the end of the list

    root = _lower(spec_dict).root
    assert root.edges[1] == AdkEdge("ocr", routes=(("ok", "guard"), ("low_quality", "abort")))


def test_function_node_kwargs(adk: AdkProgram) -> None:
    ocr = _node(adk.root, "ocr")

    assert ocr.factory == "ocr_extract"
    assert ocr.needs_deps is True
    assert dict(ocr.kwargs) == {
        "server": "ocr",
        "tool": "extract_text",
        "min_mean_confidence": 0.5,
        "max_image_bytes": 8388608,
    }
    assert dict(_node(adk.root, "schedule").kwargs) == {"service": "scheduler"}
    assert _node(adk.root, "gate").needs_deps is False


def test_retries(adk: AdkProgram) -> None:
    extraction = _extraction(adk)

    assert _node(adk.root, "ocr").retry == RetryIR(2, 0.5, 2.0)
    assert _node(adk.root, "ocr").timeout_s == 45.0
    # The agent inherits retry and timeout from its llm_extract node; the input node does not.
    agent = _node(extraction, "extractor_agent")
    assert (agent.retry, agent.timeout_s) == (RetryIR(1, 0.5, 2.0), 30.0)
    assert _node(extraction, "llm_extract_input").retry is None
    assert _node(adk.root, "gate").retry is None


def test_root_nodes_in_topological_order(adk: AdkProgram) -> None:
    assert [n.var for n in adk.root.nodes] == [
        "ocr",
        "guard",
        "extraction",
        "resolve",
        "gate",
        "review",
        "schedule",
        "present",
        "abort",
    ]


def test_topological_order_ignores_spec_order(spec_dict: SpecDict) -> None:
    nodes = spec_dict["workflow"]["nodes"]
    nodes.insert(0, nodes.pop(4))  # gate listed first

    order = [n.var for n in _lower(spec_dict).root.nodes]
    assert order.index("gate") > order.index("resolve")


def test_leaf_and_workflow_emission_order(adk: AdkProgram) -> None:
    assert [n.var for n in adk.leaf_nodes()] == [
        "ocr",
        "guard",
        "llm_extract_input",
        "extractor_agent",
        "lexical_extract",
        "merge",
        "reconcile",
        "resolve",
        "gate",
        "review",
        "schedule",
        "present",
        "abort",
    ]
    assert [n.var for n in adk.subworkflows()] == ["extraction"]


def test_nested_workflow_is_named_after_the_node(spec_dict: SpecDict) -> None:
    spec_dict["workflow"]["nodes"][2]["id"] = "extract_step"
    for edge in spec_dict["workflow"]["edges"]:
        for end in ("from", "to"):
            if edge[end] == "extraction":
                edge[end] = "extract_step"

    [workflow] = _lower(spec_dict).subworkflows()
    assert workflow.var == "extract_step"
    assert workflow.subgraph is not None
    assert workflow.subgraph.name == "extract_step"


def test_program_flags(adk: AdkProgram) -> None:
    assert adk.uses("agent")
    assert adk.uses("join")
    assert adk.uses_retry_config
    assert adk.needs_resumability
    assert adk.root.name == "full_agent"
    assert len(adk.ir_digest) == 64
    assert "extraction" in adk.node_names()


def test_lowered_program_is_immutable(adk: AdkProgram) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        adk.root = adk.root  # type: ignore[misc]
    with pytest.raises(TypeError):
        _node(adk.root, "ocr").kwargs["tool"] = "x"  # type: ignore[index]
