import socket
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk import Workflow
from google.adk.models import LlmRequest, LlmResponse
from google.adk.workflow import FunctionNode, RetryConfig
from google.genai import types
from pydantic import ValidationError

from agent_runtime import nodes, schemas
from agent_runtime.bootstrap import HttpServiceConfig, McpServerConfig, RuntimeDeps
from agent_runtime.guardrails import pii_tripwire
from agent_runtime.models import resolve_model
from agent_runtime.nodes import _not_implemented

pytestmark = pytest.mark.unit

OCR = McpServerConfig(
    url_env="OCR_MCP_URL",
    auth_token_env="MCP_AUTH_TOKEN",  # noqa: S106 - env var name
    allowed_tools=("extract_text",),
    connect_timeout_s=5.0,
    read_timeout_s=40.0,
)
SCHEDULER = HttpServiceConfig(
    base_url_env="SCHEDULER_API_URL", api_key_env="SCHEDULER_API_KEY", timeout_s=10.0
)


@pytest.fixture
def deps(monkeypatch: pytest.MonkeyPatch) -> RuntimeDeps:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("building the graph must not open connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return RuntimeDeps.lazy(mcp_servers={"ocr": OCR}, http_services={"scheduler": SCHEDULER})


def _all_nodes(deps: RuntimeDeps) -> list[FunctionNode]:
    retry = RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=4.0)
    mcp = {"deps": deps, "server": "ocr", "tool": "extract_text", "retry": retry, "timeout": 45.0}
    return [
        nodes.ocr_extract(name="ocr", **mcp),
        nodes.untrusted_text_guard(name="guard", max_chars=8000, block_at="high"),
        nodes.llm_input(name="llm_extract_input"),
        nodes.lexical_extract(name="lexical_extract", **mcp),
        nodes.reconcile(name="reconcile", sources={"llm": "a", "lexical": "b"}),
        nodes.rag_resolve(name="resolve", **mcp),
        nodes.confidence_gate(
            name="gate", accept_threshold=0.85, review_threshold=0.6, min_margin=0.05
        ),
        nodes.human_review(name="review"),
        nodes.schedule(name="schedule", deps=deps, service="scheduler", retry=retry, timeout=15.0),
        nodes.present(name="present", format="json"),
        nodes.abort(name="abort"),
    ]


def test_every_factory_is_exported() -> None:
    assert sorted(nodes.__all__) == sorted(
        [
            "ocr_extract",
            "untrusted_text_guard",
            "llm_input",
            "lexical_extract",
            "reconcile",
            "rag_resolve",
            "confidence_gate",
            "human_review",
            "schedule",
            "present",
            "abort",
        ]
    )


def test_factories_return_named_function_nodes(deps: RuntimeDeps) -> None:
    built = _all_nodes(deps)

    assert all(isinstance(node, FunctionNode) for node in built)
    assert [node.name for node in built][:3] == ["ocr", "guard", "llm_extract_input"]


def test_retry_and_timeout_reach_the_node(deps: RuntimeDeps) -> None:
    ocr = _all_nodes(deps)[0]

    assert ocr.timeout == 45.0
    assert ocr.retry_config is not None
    assert ocr.retry_config.max_attempts == 3


def test_only_human_review_reruns_on_resume(deps: RuntimeDeps) -> None:
    rerun = [node.name for node in _all_nodes(deps) if node.rerun_on_resume]

    assert rerun == ["review"]


def test_stub_nodes_build_a_workflow(deps: RuntimeDeps) -> None:
    ocr, guard, *_, present, abort = _all_nodes(deps)

    workflow = Workflow(
        name="skeleton",
        edges=[
            ("START", ocr),
            (ocr, {"ok": guard, "low_quality": abort}),
            (guard, {"clean": present, "blocked": abort}),
        ],
    )
    assert workflow.name == "skeleton"


async def test_stub_function_raises_until_phase_06() -> None:
    with pytest.raises(NotImplementedError, match="phase 06"):
        await _not_implemented("ocr")(MagicMock(), None)


def test_runtime_deps_records_configuration(deps: RuntimeDeps) -> None:
    assert deps.mcp_servers["ocr"] is OCR
    assert deps.http_services["scheduler"].timeout_s == 10.0
    with pytest.raises(TypeError):
        deps.mcp_servers["rag"] = OCR  # type: ignore[index]


def test_runtime_deps_defaults_to_empty() -> None:
    empty = RuntimeDeps.lazy()

    assert dict(empty.mcp_servers) == {}
    assert dict(empty.http_services) == {}


def test_resolve_model_passes_name_through() -> None:
    assert resolve_model("gemini-flash-latest") == "gemini-flash-latest"


def test_llm_mentions_is_the_agent_output_schema() -> None:
    assert schemas.LlmMentions is schemas.ExtractedExams


def test_schemas_are_strict() -> None:
    with pytest.raises(ValidationError):
        schemas.RunInput.model_validate({"image_path": "a.png", "raw_text": "x"})
    with pytest.raises(ValidationError):
        schemas.Candidate(code="X-1", name="Hemograma", score=0.9, matched_on="name")
    with pytest.raises(ValidationError):
        schemas.ExtractedExams.model_validate(
            {"exams": [{"name": "Hemograma", "evidence": "HMG"}] * 31}
        )


def test_schemas_are_frozen() -> None:
    item = schemas.RunInput(image_path="a.png")

    with pytest.raises(ValidationError):
        item.image_path = "b.png"  # type: ignore[misc]


@pytest.mark.xfail(
    strict=True, reason="pii_tripwire is a no-op until phase 06; remove this marker then"
)
def test_pii_tripwire_blocks_valid_cpf() -> None:
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="CPF 529.982.247-25")])]
    )

    assert isinstance(pii_tripwire(MagicMock(), request), LlmResponse)
