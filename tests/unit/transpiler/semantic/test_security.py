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
    "E_LLM_NOT_GUARDED": "/subworkflows/extraction/nodes/0",
    "E_SCHEDULE_NOT_GATED": "/workflow/nodes/6",
    "E_UNTRUSTED_AGENT_HAS_TOOLS": "/llm_agents/extractor_agent/tools",
    "E_AGENT_TOOLS_UNSUPPORTED": "/llm_agents/extractor_agent/tools",
    "E_PII_POLICY": "/policies/pii/fail_closed",
}


@pytest.mark.parametrize(("code", "pointer"), CORPUS.items())
def test_invalid_corpus(
    check_corpus: Callable[[str], list[Diagnostic]], code: str, pointer: str
) -> None:
    assert (code, pointer) in _found(check_corpus(code))


def test_valid_spec_is_guarded_and_gated(spec_dict: SpecDict, check: Check) -> None:
    assert check(spec_dict) == []


def test_side_path_around_guard(spec_dict: SpecDict, check: Check) -> None:
    # A second guard on another branch does not help: every path must be guarded.
    spec_dict["workflow"]["nodes"][0]["kind"] = "untrusted_text_guard"
    for key in ("server", "tool", "retry", "timeout_s"):
        del spec_dict["workflow"]["nodes"][0][key]
    spec_dict["workflow"]["nodes"][0]["params"] = {}
    spec_dict["workflow"]["edges"][0] = {"from": "START", "to": "extraction"}
    spec_dict["workflow"]["edges"][1:3] = []

    found = _found(_codes(check(spec_dict), Code.E_LLM_NOT_GUARDED))
    assert found == {("E_LLM_NOT_GUARDED", "/subworkflows/extraction/nodes/0")}


def test_guard_inside_subworkflow_counts(spec_dict: SpecDict, check: Check) -> None:
    # Move the guard into the extraction subworkflow, right after its START.
    root = spec_dict["workflow"]
    root["nodes"] = [n for n in root["nodes"] if n["id"] != "guard"]
    root["edges"] = [e for e in root["edges"] if e["from"] != "guard"]
    root["edges"][1]["to"] = "extraction"
    sub = spec_dict["subworkflows"]["extraction"]
    sub["nodes"].insert(0, {"id": "guard", "kind": "untrusted_text_guard"})
    sub["nodes"].append({"id": "blocked", "kind": "abort"})
    sub["edges"][0:2] = [
        {"from": "START", "to": "guard"},
        {"from": "guard", "to": "llm_extract", "route": "clean"},
        {"from": "guard", "to": "lexical_extract", "route": "clean"},
        {"from": "guard", "to": "blocked", "route": "blocked"},
    ]

    assert _codes(check(spec_dict), Code.E_LLM_NOT_GUARDED) == []


def test_review_path_still_needs_gate(spec_dict: SpecDict, check: Check) -> None:
    # resolve → review bypasses the gate even though review comes "after" it elsewhere.
    spec_dict["workflow"]["edges"].append({"from": "resolve", "to": "review"})

    assert ("E_SCHEDULE_NOT_GATED", "/workflow/nodes/6") in _found(check(spec_dict))


def test_trusted_agent_without_tools_is_fine(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["llm_agents"]["extractor_agent"]["input_trust"] = "trusted"

    assert check(spec_dict) == []


def test_pii_policy_is_unconditional(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["policies"]["pii"]["fail_closed"] = False

    [diagnostic] = check(spec_dict)
    assert diagnostic.code is Code.E_PII_POLICY


def test_subworkflow_cycle_does_not_hang(check_corpus: Callable[[str], list[Diagnostic]]) -> None:
    assert Code.E_SUBWORKFLOW_CYCLE in {d.code for d in check_corpus("E_SUBWORKFLOW_CYCLE")}
