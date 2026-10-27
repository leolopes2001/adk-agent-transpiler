import json

import pytest
from google.genai import _transformers
from pydantic import ValidationError

from agent_runtime.schemas import ExtractedExams

pytestmark = pytest.mark.unit


def test_llm_output_schema_has_no_additional_properties() -> None:
    schema = ExtractedExams.model_json_schema()

    assert "additionalProperties" not in json.dumps(schema)


def test_genai_schema_conversion_drops_nothing_else() -> None:
    converted = _transformers.t_schema(None, ExtractedExams)  # type: ignore[arg-type]

    assert converted is not None
    assert converted.additional_properties is None
    assert converted.properties is not None
    item = converted.properties["exams"].items
    assert item is not None
    assert item.additional_properties is None
    assert item.required == ["name", "evidence"]
    assert converted.properties["exams"].max_items == 30


def test_llm_output_validation_still_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractedExams.model_validate({"exams": [], "code": "LAB-10001"})
    with pytest.raises(ValidationError):
        ExtractedExams.model_validate(
            {"exams": [{"name": "TSH", "evidence": "TSH", "code": "LAB-10043"}]}
        )
