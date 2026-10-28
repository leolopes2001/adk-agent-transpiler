import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from scheduler_api import openapi
from scheduler_api.api import docs
from scheduler_api.domain.catalog import Catalog
from scheduler_api.main import API_VERSION
from scheduler_api.schemas import AppointmentCreate

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[3]
PROBLEM_JSON = "application/problem+json"
V1_OPERATIONS = [
    ("/v1/appointments", "post"),
    ("/v1/appointments", "get"),
    ("/v1/appointments/{appointment_id}", "get"),
    ("/v1/exams/{code}", "get"),
]


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(openapi.render())
    return loaded


def _validator(schema: dict[str, Any], component: str) -> Draft202012Validator:
    # Refs are relative to the document root, so the components travel along.
    return Draft202012Validator(
        {"$ref": f"#/components/schemas/{component}", "components": schema["components"]}
    )


def test_export_writes_the_document(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "openapi.json"

    assert openapi.main([str(output)]) == 0
    assert output.read_text(encoding="utf-8") == openapi.render()


def test_export_requires_an_output_path() -> None:
    assert openapi.main([]) == 1


def test_info_and_tags(schema: dict[str, Any]) -> None:
    assert schema["info"]["title"] == "Lab Exam Scheduler API"
    assert schema["info"]["version"] == API_VERSION == "1.0.0"
    assert "Authorize" in schema["info"]["description"]
    assert [tag["name"] for tag in schema["tags"]] == ["appointments", "exams", "health"]


def test_v1_operations_require_the_api_key(schema: dict[str, Any]) -> None:
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
    for path, method in V1_OPERATIONS:
        assert schema["paths"][path][method]["security"] == [{"APIKeyHeader": []}], path
    for path in ("/health/live", "/health/ready"):
        assert "security" not in schema["paths"][path]["get"]


def test_post_declares_a_required_idempotency_key(schema: dict[str, Any]) -> None:
    parameters = schema["paths"]["/v1/appointments"]["post"]["parameters"]
    [key] = [p for p in parameters if p["name"] == "Idempotency-Key"]

    assert key["in"] == "header"
    assert key["required"] is True
    assert key["schema"]["minLength"] == 8
    assert key["schema"]["maxLength"] == 128
    assert key["schema"]["pattern"]
    assert key["description"]
    assert key["examples"]["example"]["value"] == docs.IDEMPOTENCY_KEY_EXAMPLE


def test_request_id_header_is_documented(schema: dict[str, Any]) -> None:
    for path, method in V1_OPERATIONS:
        parameters = schema["paths"][path][method]["parameters"]
        [request_id] = [p for p in parameters if p["name"] == "X-Request-ID"]
        assert request_id["required"] is False


def test_post_documents_every_status(schema: dict[str, Any]) -> None:
    responses = schema["paths"]["/v1/appointments"]["post"]["responses"]

    assert set(responses) == {"200", "201", "400", "401", "409", "422", "500"}
    assert set(responses["201"]["headers"]) == {"Location", "X-Request-ID"}
    assert "Idempotent-Replayed" in responses["200"]["headers"]
    for status in ("200", "201"):
        content = responses[status]["content"]["application/json"]
        assert content["schema"] == {"$ref": "#/components/schemas/AppointmentOut"}


def test_errors_are_problem_json_with_examples(schema: dict[str, Any]) -> None:
    validator = _validator(schema, "Problem")
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for status, response in operation["responses"].items():
                if int(status) < 400 or path.startswith("/health"):
                    continue
                assert set(response["content"]) == {PROBLEM_JSON}, (path, method, status)
                content = response["content"][PROBLEM_JSON]
                assert content["schema"] == {"$ref": "#/components/schemas/Problem"}
                assert content["examples"], (path, method, status)
                for example in content["examples"].values():
                    validator.validate(example["value"])
                    assert example["value"]["status"] == int(status)


def test_post_documents_the_catalog_and_conflict_problems(schema: dict[str, Any]) -> None:
    responses = schema["paths"]["/v1/appointments"]["post"]["responses"]

    assert set(responses["409"]["content"][PROBLEM_JSON]["examples"]) == {"idempotency_key_reuse"}
    assert set(responses["422"]["content"][PROBLEM_JSON]["examples"]) == {
        "unknown_exam_code",
        "exam_name_mismatch",
        "validation_error",
    }


def test_no_default_validation_error_schema(schema: dict[str, Any]) -> None:
    assert "HTTPValidationError" not in schema["components"]["schemas"]


def test_body_examples_match_the_contract(schema: dict[str, Any]) -> None:
    body = schema["paths"]["/v1/appointments"]["post"]["requestBody"]
    examples = body["content"]["application/json"]["examples"]
    validator = _validator(schema, "AppointmentCreate")

    assert set(examples) == {"two_exams", "unknown_code", "personal_data"}
    validator.validate(examples["two_exams"]["value"])
    validator.validate(examples["unknown_code"]["value"])
    assert not validator.is_valid(examples["personal_data"]["value"])
    AppointmentCreate.model_validate(examples["two_exams"]["value"])
    with pytest.raises(ValidationError, match="patient_name"):
        AppointmentCreate.model_validate(examples["personal_data"]["value"])


def test_valid_example_uses_real_catalog_entries() -> None:
    catalog = Catalog.load(ROOT / "data" / "exams_catalog.json")

    for exam in docs.CREATE_EXAMPLES["two_exams"]["value"]["exams"]:
        assert catalog.name_matches(exam["code"], exam["name"]), exam
    for exam in docs.CREATE_EXAMPLES["unknown_code"]["value"]["exams"]:
        assert catalog.get(exam["code"]) is None


def test_appointment_example_matches_the_response_schema(schema: dict[str, Any]) -> None:
    _validator(schema, "AppointmentOut").validate(docs.APPOINTMENT_EXAMPLE)


def test_contract_has_no_personal_data_fields(schema: dict[str, Any]) -> None:
    create = schema["components"]["schemas"]["AppointmentCreate"]

    assert create["additionalProperties"] is False
    assert set(create["properties"]) == {
        "patient_ref",
        "exams",
        "source_document_sha256",
        "requested_by",
    }
    assert schema["components"]["schemas"]["Exam"]["additionalProperties"] is False
