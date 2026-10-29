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
    assert ("E_TIMEOUT_BUDGET", "/workflow/nodes/3/timeout_s") in _found(
        check_corpus("E_TIMEOUT_BUDGET")
    )


def test_message(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][3]["timeout_s"] = 10

    [diagnostic] = check(spec_dict)
    assert diagnostic.message == (
        "timeout_s do nó (10s) é menor que mcp_servers.rag.read_timeout_s (15s)"
    )
    assert diagnostic.hint == "use timeout_s >= 15"


def test_equal_timeout_is_enough(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][3]["timeout_s"] = 15

    assert check(spec_dict) == []


def test_default_timeout_counts(spec_dict: SpecDict, check: Check) -> None:
    del spec_dict["workflow"]["nodes"][0]["timeout_s"]  # default 30 < ocr read 40

    [diagnostic] = check(spec_dict)
    assert diagnostic.code is Code.E_TIMEOUT_BUDGET
    assert diagnostic.pointer == "/workflow/nodes/0"
    assert diagnostic.message.startswith("o timeout padrão do nó (30s)")


def test_http_service_budget(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][6]["timeout_s"] = 5

    assert _found(check(spec_dict)) == {("E_TIMEOUT_BUDGET", "/workflow/nodes/6/timeout_s")}


def test_unknown_server_is_not_a_budget_error(spec_dict: SpecDict, check: Check) -> None:
    spec_dict["workflow"]["nodes"][3]["server"] = "nope"

    assert _codes(check(spec_dict), Code.E_TIMEOUT_BUDGET) == []
