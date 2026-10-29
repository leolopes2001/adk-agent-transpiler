import typing

import pytest
from pydantic import BaseModel

from transpiler.registry import REGISTRY, NodeKind, PortType, kind_spec
from transpiler.spec import models

pytestmark = pytest.mark.unit

REFERENCE_FIELDS = ("server", "tool", "service", "agent", "ref")


def _node_models() -> dict[str, type[BaseModel]]:
    union = typing.get_args(models.NodeSpec)[0]
    result: dict[str, type[BaseModel]] = {}
    for model in typing.get_args(union):
        [kind] = typing.get_args(model.model_fields["kind"].annotation)
        result[kind] = model
    return result


NODE_MODELS = _node_models()


def test_thirteen_kinds() -> None:
    assert len(REGISTRY) == 13
    assert set(REGISTRY) == set(NodeKind)


def test_every_spec_kind_is_registered() -> None:
    assert set(NODE_MODELS) == {kind.value for kind in NodeKind}


@pytest.mark.parametrize("kind", list(NodeKind))
def test_requires_matches_node_model(kind: NodeKind) -> None:
    fields = NODE_MODELS[kind].model_fields
    required = {name for name in REFERENCE_FIELDS if name in fields and fields[name].is_required()}

    assert REGISTRY[kind].requires == required


@pytest.mark.parametrize("kind", list(NodeKind))
def test_params_model_matches_node_model(kind: NodeKind) -> None:
    params = NODE_MODELS[kind].model_fields.get("params")

    assert REGISTRY[kind].params_model is (None if params is None else params.annotation)


@pytest.mark.parametrize("kind", list(NodeKind))
def test_accepts_retry_matches_node_model(kind: NodeKind) -> None:
    fields = NODE_MODELS[kind].model_fields

    assert REGISTRY[kind].accepts_retry == ("retry" in fields and "timeout_s" in fields)


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda spec: spec.kind)
def test_output_shape_is_consistent(spec: typing.Any) -> None:
    if spec.terminal:
        assert spec.output_type is None
        assert not spec.routes
    if spec.is_router:
        assert spec.output_type is None


def test_routers() -> None:
    routers = {kind for kind, spec in REGISTRY.items() if spec.is_router}

    assert routers == {
        NodeKind.OCR_EXTRACT,
        NodeKind.UNTRUSTED_TEXT_GUARD,
        NodeKind.CONFIDENCE_GATE,
        NodeKind.HUMAN_REVIEW,
    }
    assert set(REGISTRY[NodeKind.CONFIDENCE_GATE].routes) == {"accepted", "review", "empty"}


def test_terminals() -> None:
    terminals = {kind for kind, spec in REGISTRY.items() if spec.terminal}

    assert terminals == {NodeKind.PRESENT, NodeKind.ABORT}


def test_function_kinds_have_distinct_factories() -> None:
    factories = [spec.factory for spec in REGISTRY.values() if spec.factory is not None]

    assert len(factories) == len(set(factories)) == 11
    assert kind_spec("llm_extract").factory == "llm_input"


def test_primitives() -> None:
    by_primitive = {spec.kind: spec.adk_primitive for spec in REGISTRY.values()}

    assert by_primitive[NodeKind.LLM_EXTRACT] == "agent"
    assert by_primitive[NodeKind.JOIN] == "join"
    assert by_primitive[NodeKind.SUBWORKFLOW] == "workflow"
    assert sum(1 for primitive in by_primitive.values() if primitive == "function") == 10


def test_only_human_review_reruns_on_resume() -> None:
    assert [spec.kind for spec in REGISTRY.values() if spec.rerun_on_resume] == [
        NodeKind.HUMAN_REVIEW
    ]


def test_reconcile_joins_llm_and_lexical_mentions() -> None:
    spec = kind_spec("reconcile")

    assert spec.input_types == {PortType.JOIN_RESULT}
    assert set(spec.join_inputs) == {PortType.LLM_MENTIONS, PortType.EXAM_MENTIONS}


def test_output_types() -> None:
    assert kind_spec("confidence_gate").output_types() == {
        PortType.GATE_DECISION,
        PortType.RUN_FAILURE,
    }
    assert kind_spec("rag_resolve").output_types() == {PortType.RESOLVED_EXAMS}
    assert kind_spec("present").output_types() == frozenset()


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        REGISTRY[NodeKind.JOIN] = REGISTRY[NodeKind.ABORT]  # type: ignore[index]


def test_unknown_kind() -> None:
    with pytest.raises(ValueError, match="shell"):
        kind_spec("shell")


def test_tools_are_declared_for_kinds_that_call_a_tool() -> None:
    with_tools = {kind: spec.tools for kind, spec in REGISTRY.items() if spec.tools}

    assert with_tools == {
        NodeKind.OCR_EXTRACT: {"extract_text"},
        NodeKind.LEXICAL_EXTRACT: {"batch_search"},
        NodeKind.RAG_RESOLVE: {"batch_search"},
    }
    assert all(("tool" in spec.requires) == bool(spec.tools) for spec in REGISTRY.values())
