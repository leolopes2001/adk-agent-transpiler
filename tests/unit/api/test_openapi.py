import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from scheduler_api import openapi
from scheduler_api.api import docs
from scheduler_api.domain.catalog import Catalog
from scheduler_api.main import API_VERSION
from scheduler_api.schemas import PATIENT_REF_PATTERN, NewAppointment

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


def test_info(schema: dict[str, Any]) -> None:
    description = schema["info"]["description"]

    assert schema["info"]["title"] == "API de Agendamento de Exames"
    assert schema["info"]["version"] == API_VERSION == "1.1.0"
    for phrase in ("Authorize", "X-API-Key", "patient_ref", "dados pessoais"):
        assert phrase in description, phrase


def test_declares_the_three_tags_in_order(schema: dict[str, Any]) -> None:
    assert [tag["name"] for tag in schema["tags"]] == ["Agendamentos", "Exames", "Health"]
    for tag in schema["tags"]:
        assert tag["description"], tag["name"]
        assert "\n" not in tag["description"], tag["name"]


def _operations(schema: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            yield path, method, operation


def test_every_operation_has_exactly_one_declared_tag(schema: dict[str, Any]) -> None:
    declared = {tag["name"] for tag in schema["tags"]}
    for path, method, operation in _operations(schema):
        tags = operation.get("tags", [])
        assert len(tags) == 1, (path, method)
        assert tags[0] in declared, (path, method)


def test_every_operation_has_a_summary_and_description(schema: dict[str, Any]) -> None:
    for path, method, operation in _operations(schema):
        assert operation.get("summary"), (path, method)
        assert operation.get("description"), (path, method)


def test_exposes_only_the_contract_schemas(schema: dict[str, Any]) -> None:
    assert set(schema["components"]["schemas"]) == {
        "NewAppointment",
        "Appointment",
        "Exam",
        "CatalogEntry",
        "Problem",
        "FieldError",
        "HealthStatus",
    }


def test_v1_operations_require_the_api_key(schema: dict[str, Any]) -> None:
    assert schema["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Chave da API configurada no servidor (`SCHEDULER_API_KEY`).",
    }
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
        assert request_id["description"]
        assert re.fullmatch(request_id["schema"]["pattern"], "run-77:a.b_c")
        assert not re.fullmatch(request_id["schema"]["pattern"], "x" * 129)


def test_list_documents_limit_and_patient_ref(schema: dict[str, Any]) -> None:
    parameters = {p["name"]: p for p in schema["paths"]["/v1/appointments"]["get"]["parameters"]}
    limit = parameters["limit"]["schema"]
    patient_ref = parameters["patient_ref"]

    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 100, 20)
    assert patient_ref["required"] is False
    [example] = patient_ref["examples"].values()
    assert re.fullmatch(PATIENT_REF_PATTERN, example["value"])


def test_list_returns_an_array_of_the_post_model(schema: dict[str, Any]) -> None:
    media = schema["paths"]["/v1/appointments"]["get"]["responses"]["200"]["content"]
    response_schema = media["application/json"]["schema"]

    assert response_schema["type"] == "array"
    assert response_schema["items"] == {"$ref": "#/components/schemas/Appointment"}
    _validator(schema, "Appointment").validate(media["application/json"]["example"][0])


def test_path_examples_match_their_patterns(schema: dict[str, Any]) -> None:
    catalog = Catalog.load(ROOT / "data" / "exams_catalog.json")
    for path, method in V1_OPERATIONS:
        for parameter in schema["paths"][path][method]["parameters"]:
            if parameter["in"] != "path":
                continue
            assert parameter["examples"], (path, parameter["name"])
            for example in parameter["examples"].values():
                assert re.fullmatch(parameter["schema"]["pattern"], example["value"])
                if parameter["name"] == "code":
                    assert catalog.get(example["value"]) is not None


def test_post_documents_every_status(schema: dict[str, Any]) -> None:
    responses = schema["paths"]["/v1/appointments"]["post"]["responses"]

    assert set(responses) == {"200", "201", "400", "401", "409", "422", "500"}
    assert set(responses["201"]["headers"]) == {"Location", "X-Request-ID"}
    assert "Idempotent-Replayed" in responses["200"]["headers"]
    for status in ("200", "201"):
        content = responses[status]["content"]["application/json"]
        assert content["schema"] == {"$ref": "#/components/schemas/Appointment"}


@pytest.mark.parametrize(
    ("path", "method", "statuses"),
    [
        ("/v1/appointments", "post", {"400", "401", "409", "422", "500"}),
        ("/v1/appointments", "get", {"401", "422", "500"}),
        ("/v1/appointments/{appointment_id}", "get", {"401", "404", "422", "500"}),
        ("/v1/exams/{code}", "get", {"401", "404", "422", "500"}),
    ],
)
def test_documents_the_error_statuses(
    schema: dict[str, Any], path: str, method: str, statuses: set[str]
) -> None:
    responses = schema["paths"][path][method]["responses"]

    assert {status for status in responses if int(status) >= 400} == statuses


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
                    assert example["summary"], (path, method, status)
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
    validator = _validator(schema, "NewAppointment")

    assert set(examples) == {"two_exams", "unknown_code", "personal_data"}
    validator.validate(examples["two_exams"]["value"])
    validator.validate(examples["unknown_code"]["value"])
    assert not validator.is_valid(examples["personal_data"]["value"])
    NewAppointment.model_validate(examples["two_exams"]["value"])
    with pytest.raises(ValidationError, match="patient_name"):
        NewAppointment.model_validate(examples["personal_data"]["value"])


def test_valid_example_uses_real_catalog_entries() -> None:
    catalog = Catalog.load(ROOT / "data" / "exams_catalog.json")

    for exam in docs.CREATE_EXAMPLES["two_exams"]["value"]["exams"]:
        assert catalog.name_matches(exam["code"], exam["name"]), exam
    for exam in docs.CREATE_EXAMPLES["unknown_code"]["value"]["exams"]:
        assert catalog.get(exam["code"]) is None


def test_appointment_example_matches_the_response_schema(schema: dict[str, Any]) -> None:
    _validator(schema, "Appointment").validate(docs.APPOINTMENT_EXAMPLE)


def test_contract_has_no_personal_data_fields(schema: dict[str, Any]) -> None:
    create = schema["components"]["schemas"]["NewAppointment"]

    assert create["additionalProperties"] is False
    assert set(create["properties"]) == {
        "patient_ref",
        "exams",
        "source_document_sha256",
        "requested_by",
    }
    assert schema["components"]["schemas"]["Exam"]["additionalProperties"] is False
