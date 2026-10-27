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


def test_invalid_corpus(check_corpus: Callable[[str], list[Diagnostic]]) -> None:
    found = _found(check_corpus("E_TYPE_MISMATCH"))

    assert ("E_TYPE_MISMATCH", "/workflow/edges/11") in found
    assert ("E_TYPE_MISMATCH", "/workflow/edges/12") in found


def test_valid_spec_types(spec_dict: SpecDict, check: Check) -> None:
    assert _codes(check(spec_dict), Code.E_TYPE_MISMATCH) == []


def test_mismatch_names_route_and_types(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"][2]["to"] = "present"  # ocr low_quality → present

    [diagnostic] = _codes(check(spec_dict), Code.E_TYPE_MISMATCH)
    assert diagnostic.pointer == "/workflow/edges/2"
    assert diagnostic.message == (
        '"ocr" produz RunFailure pela rota "low_quality", mas "present" (present) '
        "espera GateDecision | ScheduleOutcome"
    )


def test_start_of_root_is_run_input(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"][0]["to"] = "guard"
    spec_dict["workflow"]["edges"].append({"from": "START", "to": "ocr"})

    found = _found(check(spec_dict))
    assert ("E_TYPE_MISMATCH", "/workflow/edges/0") in found


def test_subworkflow_output_is_its_final_node(spec_dict: SpecDict, check: Check) -> None:
    # extraction ends in reconcile (ExamMentions); gate expects ResolvedExams.
    spec_dict["workflow"]["edges"][5]["to"] = "gate"
    spec_dict["workflow"]["edges"][6] = {"from": "extraction", "to": "resolve"}

    found = _found(check(spec_dict))
    assert ("E_TYPE_MISMATCH", "/workflow/edges/5") in found


def test_subworkflow_input_comes_from_parent(spec_dict: SpecDict, check: Check) -> None:
    # Feeding the subworkflow straight from OCR skips the guard: inner nodes get OcrDocument.
    spec_dict["workflow"]["edges"][1]["to"] = "extraction"
    spec_dict["workflow"]["edges"][3]["to"] = "abort"

    found = _found(check(spec_dict))
    assert ("E_TYPE_MISMATCH", "/subworkflows/extraction/edges/0") in found
    assert ("E_TYPE_MISMATCH", "/subworkflows/extraction/edges/1") in found


def test_reconcile_needs_join_of_llm_and_lexical(spec_dict: SpecDict, check: Check) -> None:
    sub = spec_dict["subworkflows"]["extraction"]
    sub["nodes"][0] = {
        "id": "llm_extract",
        "kind": "lexical_extract",
        "server": "rag",
        "tool": "batch_search",
        "timeout_s": 20,
    }
    del spec_dict["llm_agents"]["extractor_agent"]
    del spec_dict["models"]["extractor"]

    [diagnostic] = _codes(check(spec_dict), Code.E_TYPE_MISMATCH)
    assert diagnostic.pointer == "/subworkflows/extraction/edges/4"
    assert "ExamMentions | ExamMentions" in diagnostic.message


def test_reconcile_without_join(spec_dict: SpecDict, check: Check) -> None:
    edges = spec_dict["subworkflows"]["extraction"]["edges"]
    edges[4]["from"] = "lexical_extract"
    edges[3]["to"] = "reconcile"
    spec_dict["subworkflows"]["extraction"]["nodes"].pop(2)
    del edges[2]

    found = _found(check(spec_dict))
    assert ("E_TYPE_MISMATCH", "/subworkflows/extraction/edges/2") in found


def test_present_accepts_both_decision_and_outcome(spec_dict: SpecDict, check: Check) -> None:
    # review --rejected--> present carries GateDecision; schedule → present carries ScheduleOutcome.
    assert _codes(check(spec_dict), Code.E_TYPE_MISMATCH) == []


def test_unknown_nodes_are_skipped(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["edges"].append({"from": "ghost", "to": "present"})

    assert _codes(check(spec_dict), Code.E_TYPE_MISMATCH) == []
