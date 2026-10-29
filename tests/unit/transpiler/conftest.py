import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from transpiler import semantic
from transpiler.errors import Diagnostic
from transpiler.spec.loader import load_spec, parse_spec

SPECS = Path(__file__).parents[2] / "data" / "specs"
VALID_SPECS = SPECS / "valid"
INVALID_SPECS = SPECS / "invalid"

type SpecDict = dict[str, Any]
type WriteSpec = Callable[[SpecDict], Path]
type Check = Callable[[SpecDict], list[Diagnostic]]


@pytest.fixture(scope="session")
def _full_spec() -> SpecDict:
    loaded: SpecDict = json.loads((VALID_SPECS / "full.json").read_text(encoding="utf-8"))
    return loaded


@pytest.fixture
def spec_dict(_full_spec: SpecDict) -> SpecDict:
    """A fresh, mutable copy of a valid spec that uses every node kind."""
    return copy.deepcopy(_full_spec)


@pytest.fixture
def write_spec(tmp_path: Path) -> WriteSpec:
    """Writes a spec dict next to a copy of the fixture's instruction files."""
    shutil.copytree(VALID_SPECS / "instructions", tmp_path / "instructions", dirs_exist_ok=True)

    def write(spec: SpecDict) -> Path:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    return write


@pytest.fixture
def check(tmp_path: Path, write_spec: WriteSpec) -> Check:
    """Runs the semantic pass on a spec dict (instruction files resolve under tmp_path)."""

    def run(spec: SpecDict) -> list[Diagnostic]:
        return semantic.validate(parse_spec(spec), tmp_path)

    return run


def _check_corpus(code: str) -> list[Diagnostic]:
    path = INVALID_SPECS / f"{code}.json"
    return semantic.validate(load_spec(path).spec, path.parent)


@pytest.fixture
def check_corpus() -> Callable[[str], list[Diagnostic]]:
    """Semantic diagnostics for the invalid-corpus file named after a code."""
    return _check_corpus
