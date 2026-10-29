"""Transpiler registry <-> runtime factories <-> runtime schemas.

The transpiler never imports the runtime and the runtime never imports the
transpiler, so the generated code is their only meeting point. These tests
keep the two sides in agreement: every kind's factory exists, takes exactly
the keywords the generated code passes (params, deps/server/tool/service,
retry/timeout), with the same defaults and literal choices, and every port
type name has a schema.
"""

import inspect
import typing
from typing import Any

import pytest
from google.adk.workflow import FunctionNode, RetryConfig
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from agent_runtime import nodes, schemas
from agent_runtime.bootstrap import RuntimeDeps
from agent_runtime.nodes.confidence_gate import ConfidenceGateParams
from agent_runtime.nodes.human_review import HumanReviewParams
from agent_runtime.nodes.lexical_extract import LexicalExtractParams
from agent_runtime.nodes.llm_input import LlmInputParams
from agent_runtime.nodes.ocr_extract import OcrExtractParams
from agent_runtime.nodes.present import PresentParams
from agent_runtime.nodes.rag_resolve import RagResolveParams
from agent_runtime.nodes.reconcile import ReconcileParams
from agent_runtime.nodes.untrusted_text_guard import UntrustedTextGuardParams
from transpiler.registry import REGISTRY, NodeKind, NodeKindSpec, PortType
from transpiler.spec import models as spec_models

pytestmark = pytest.mark.contract

WITH_FACTORY = [spec for spec in REGISTRY.values() if spec.factory is not None]
RUNTIME_PARAMS: dict[NodeKind, type[BaseModel]] = {
    NodeKind.OCR_EXTRACT: OcrExtractParams,
    NodeKind.UNTRUSTED_TEXT_GUARD: UntrustedTextGuardParams,
    NodeKind.LLM_EXTRACT: LlmInputParams,
    NodeKind.LEXICAL_EXTRACT: LexicalExtractParams,
    NodeKind.RECONCILE: ReconcileParams,
    NodeKind.RAG_RESOLVE: RagResolveParams,
    NodeKind.CONFIDENCE_GATE: ConfidenceGateParams,
    NodeKind.HUMAN_REVIEW: HumanReviewParams,
    NodeKind.PRESENT: PresentParams,
}
LOWERING_KEYWORDS = {NodeKind.RECONCILE: {"sources"}}
"""Keywords the lowering adds that are not spec params."""


def _ids(spec: NodeKindSpec) -> str:
    return spec.kind.value


def _expected_keywords(spec: NodeKindSpec) -> set[str]:
    params = set(spec.params_model.model_fields) if spec.params_model else set()
    if spec.kind is NodeKind.LLM_EXTRACT:
        # The factory builds the <id>_input node; the Agent is emitted inline.
        return {"name"} | params
    io = {"retry", "timeout"} if spec.accepts_retry else set()
    requires = set()
    if spec.requires & {"server", "service"}:
        requires.add("deps")
    requires |= {r for r in spec.requires if r in {"server", "tool", "service"}}
    return {"name"} | params | io | requires | LOWERING_KEYWORDS.get(spec.kind, set())


def _factory(spec: NodeKindSpec) -> Any:
    assert spec.factory is not None
    return getattr(nodes, spec.factory)


@pytest.mark.parametrize("spec", WITH_FACTORY, ids=_ids)
def test_factory_exists_and_is_exported(spec: NodeKindSpec) -> None:
    assert spec.factory in nodes.__all__
    assert callable(_factory(spec))


@pytest.mark.parametrize("spec", WITH_FACTORY, ids=_ids)
def test_factory_keywords_are_what_generated_code_passes(spec: NodeKindSpec) -> None:
    parameters = inspect.signature(_factory(spec)).parameters

    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert set(parameters) == _expected_keywords(spec)


def _spec_default(field: FieldInfo) -> Any:
    if field.default_factory is not None:
        return field.default_factory()  # type: ignore[call-arg]
    return field.default


@pytest.mark.parametrize("spec", [s for s in WITH_FACTORY if s.params_model], ids=_ids)
def test_factory_defaults_match_the_spec_params(spec: NodeKindSpec) -> None:
    assert spec.params_model is not None
    parameters = inspect.signature(_factory(spec)).parameters

    for name, field in spec.params_model.model_fields.items():
        expected = _spec_default(field)
        if field.is_required():
            assert parameters[name].default is inspect.Parameter.empty, name
        else:
            assert parameters[name].default == expected, name


@pytest.mark.parametrize("spec", [s for s in WITH_FACTORY if s.params_model], ids=_ids)
def test_runtime_params_model_mirrors_the_spec_params(spec: NodeKindSpec) -> None:
    assert spec.params_model is not None
    runtime = RUNTIME_PARAMS[spec.kind]
    extra = LOWERING_KEYWORDS.get(spec.kind, set())

    assert set(runtime.model_fields) == set(spec.params_model.model_fields) | extra
    for name, field in spec.params_model.model_fields.items():
        mine = runtime.model_fields[name]
        assert mine.is_required() == field.is_required(), name
        if not field.is_required():
            assert mine.default == _spec_default(field), name
        if typing.get_origin(field.annotation) is typing.Literal:
            assert typing.get_args(mine.annotation) == typing.get_args(field.annotation), name


@pytest.mark.parametrize("spec", [s for s in WITH_FACTORY if s.params_model], ids=_ids)
def test_runtime_accepts_the_spec_defaults(spec: NodeKindSpec) -> None:
    assert spec.params_model is not None
    required = {
        "accept_threshold": 0.85,
        "review_threshold": 0.6,
        "min_margin": 0.05,
    }
    values = {
        name: required[name] if field.is_required() else _spec_default(field)
        for name, field in spec.params_model.model_fields.items()
    }
    values |= {"sources": {"llm": "a", "lexical": "b"}} if spec.kind is NodeKind.RECONCILE else {}

    RUNTIME_PARAMS[spec.kind].model_validate(values)


def _build(spec: NodeKindSpec) -> FunctionNode:
    kwargs: dict[str, Any] = {"name": f"n_{spec.kind.value}"}
    keywords = _expected_keywords(spec)
    if "deps" in keywords:
        kwargs["deps"] = RuntimeDeps.lazy()
    kwargs |= {k: "x" for k in ("server", "tool", "service") if k in keywords}
    if "retry" in keywords:
        kwargs |= {"retry": RetryConfig(max_attempts=2), "timeout": 5.0}
    if spec.kind is NodeKind.CONFIDENCE_GATE:
        kwargs |= {"accept_threshold": 0.85, "review_threshold": 0.6, "min_margin": 0.05}
    if spec.kind is NodeKind.RECONCILE:
        kwargs["sources"] = {"llm": "a", "lexical": "b"}
    node: FunctionNode = _factory(spec)(**kwargs)
    return node


@pytest.mark.parametrize("spec", WITH_FACTORY, ids=_ids)
def test_node_flags_match_the_registry(spec: NodeKindSpec) -> None:
    node = _build(spec)

    assert isinstance(node, FunctionNode)
    assert node.rerun_on_resume is spec.rerun_on_resume
    # llm_extract's retry and timeout go to the Agent, not to its input node.
    takes_retry = spec.accepts_retry and spec.kind is not NodeKind.LLM_EXTRACT
    assert (node.retry_config is not None) is takes_retry


@pytest.mark.parametrize("port", list(PortType), ids=lambda p: p.value)
def test_every_port_type_has_a_schema(port: PortType) -> None:
    schema = getattr(schemas, port.value)

    if port is PortType.JOIN_RESULT:
        assert isinstance(schema, typing.TypeAliasType)
    else:
        assert issubclass(schema, BaseModel)


def test_llm_output_schemas_allowed_by_the_spec_exist() -> None:
    allowed = typing.get_args(spec_models.LlmAgentSpec.model_fields["output_schema"].annotation)

    for name in allowed:
        assert issubclass(getattr(schemas, name), BaseModel)


def test_tools_the_registry_promises_are_the_server_tools() -> None:
    tools = {spec.kind: spec.tools for spec in REGISTRY.values() if spec.tools}

    assert tools == {
        NodeKind.OCR_EXTRACT: frozenset({"extract_text"}),
        NodeKind.LEXICAL_EXTRACT: frozenset({"batch_search"}),
        NodeKind.RAG_RESOLVE: frozenset({"batch_search"}),
    }
