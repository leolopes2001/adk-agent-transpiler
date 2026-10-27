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
    "E_UNKNOWN_REF": "/workflow/nodes/3/server",
    "E_TOOL_NOT_ALLOWED": "/workflow/nodes/3/tool",
    "E_AGENT_REUSED": "/subworkflows/extraction/nodes/4/agent",
    "E_INSTRUCTION_FILE": "/llm_agents/extractor_agent/instruction_file",
    "W_UNUSED_DECLARATION": "/models/spare",
}


@pytest.mark.parametrize(("code", "pointer"), CORPUS.items())
def test_invalid_corpus(
    check_corpus: Callable[[str], list[Diagnostic]], code: str, pointer: str
) -> None:
    assert (code, pointer) in _found(check_corpus(code))


@pytest.mark.parametrize(
    ("path", "field", "pointer"),
    [
        (("workflow", "nodes", 0), "server", "/workflow/nodes/0/server"),
        (("workflow", "nodes", 6), "service", "/workflow/nodes/6/service"),
        (("workflow", "nodes", 2), "ref", "/workflow/nodes/2/ref"),
        (
            ("subworkflows", "extraction", "nodes", 0),
            "agent",
            "/subworkflows/extraction/nodes/0/agent",
        ),
        (("llm_agents", "extractor_agent"), "model", "/llm_agents/extractor_agent/model"),
    ],
)
def test_unknown_reference(
    spec_dict: SpecDict, check: Check, path: tuple[str | int, ...], field: str, pointer: str
) -> None:
    target: Any = spec_dict
    for part in path:
        target = target[part]
    target[field] = "missing_thing"

    found = _found(check(spec_dict))
    assert ("E_UNKNOWN_REF", pointer) in found


def test_unknown_reference_hint(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][0]["server"] = "vision"

    [diagnostic] = _codes(check(spec_dict), Code.E_UNKNOWN_REF)
    assert diagnostic.message == 'servidor MCP "vision" não está declarado'
    assert diagnostic.hint == "declarados: ocr, rag"


def test_tool_not_allowed_message(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][3]["tool"] = "search_exams"

    [diagnostic] = _codes(check(spec_dict), Code.E_TOOL_NOT_ALLOWED)
    assert diagnostic.message == (
        "A tool \"search_exams\" não está em mcp_servers.rag.allowed_tools ['batch_search']."
    )


def test_instruction_file_missing(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["llm_agents"]["extractor_agent"]["instruction_file"] = "instructions/nope.md"

    assert _found(check(spec_dict)) == {
        ("E_INSTRUCTION_FILE", "/llm_agents/extractor_agent/instruction_file")
    }


def test_unused_declarations(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["mcp_servers"]["spare"] = dict(spec_dict["mcp_servers"]["rag"])
    spec_dict["mcp_servers"]["rag"]["allowed_tools"] = ["batch_search", "search_exams"]
    spec_dict["http_services"]["billing"] = dict(spec_dict["http_services"]["scheduler"])
    spec_dict["llm_agents"]["idle"] = dict(spec_dict["llm_agents"]["extractor_agent"])
    spec_dict["subworkflows"]["unused"] = {
        "nodes": [{"id": "only", "kind": "abort"}],
        "edges": [{"from": "START", "to": "only"}],
    }

    diagnostics = check(spec_dict)
    assert {d.pointer for d in _codes(diagnostics, Code.W_UNUSED_DECLARATION)} == {
        "/mcp_servers/spare",
        "/mcp_servers/rag/allowed_tools/1",
        "/http_services/billing",
        "/llm_agents/idle",
        "/subworkflows/unused",
    }
    assert all(d.code is Code.W_UNUSED_DECLARATION for d in diagnostics)


def test_model_of_unused_agent_is_unused(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["models"]["other"] = dict(spec_dict["models"]["extractor"])
    spec_dict["llm_agents"]["idle"] = {
        **spec_dict["llm_agents"]["extractor_agent"],
        "model": "other",
    }

    pointers = {d.pointer for d in _codes(check(spec_dict), Code.W_UNUSED_DECLARATION)}
    assert pointers == {"/llm_agents/idle", "/models/other"}
