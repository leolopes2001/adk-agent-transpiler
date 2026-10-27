import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

VALID_SPECS = Path(__file__).parents[2] / "data" / "specs" / "valid"

type SpecDict = dict[str, Any]
type WriteSpec = Callable[[SpecDict], Path]


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
