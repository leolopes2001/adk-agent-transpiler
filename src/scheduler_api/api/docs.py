from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from scheduler_api.api.errors import (
    IDEMPOTENCY_KEY_REUSE_DETAIL,
    INTERNAL_ERROR_DETAIL,
    INVALID_API_KEY_DETAIL,
    INVALID_IDEMPOTENCY_KEY_DETAIL,
    PROBLEM_MEDIA_TYPE,
    PROBLEM_TITLES,
    Problem,
    problem_type,
)
from scheduler_api.api.middleware import REQUEST_ID_HEADER

DESCRIPTION = """
Records **exam scheduling requests** sent by the exam scheduling agent. A request
is stored with status `requested`; booking a time slot is out of scope.

### Flow
1. Click **Authorize** and enter the API key (`X-API-Key` header).
2. `POST /v1/appointments` with an `Idempotency-Key` header and the request body.
   Every exam code must exist in the catalog and its name must match the
   catalog name (case, accents and punctuation are ignored).
3. Sending the same request again with the same key returns the original
   appointment (`200` with `Idempotent-Replayed: true`) instead of creating
   another one. The same key with a different body is a `409`. Keys expire
   after `IDEMPOTENCY_TTL_HOURS` (24 by default); after that the same request
   creates a new appointment.
4. `GET /v1/appointments/{appointment_id}` (the `Location` of the `201`) reads it back.

The contract has no field for personal data: the patient is identified only by
`patient_ref`, an opaque pseudonym, and unknown fields are rejected with `422`.

### Errors
Every error is `application/problem+json` (RFC 9457) with a `type` URI per error
code, the `request_id` (same as the `X-Request-ID` response header) and, for
validation errors, an `errors` list with the location of each failing field.

### Example
```
curl -X POST http://localhost:8000/v1/appointments \\
  -H "X-API-Key: $SCHEDULER_API_KEY" \\
  -H "Idempotency-Key: 3f0c9a8e5b7d4c21a6e9f0b1c2d3e4f5" \\
  -H "Content-Type: application/json" \\
  -d '{"patient_ref": "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA",
       "exams": [{"code": "LAB-10001", "name": "Hemograma completo"}],
       "source_document_sha256": "9b1e2c4d6f8a0b1c3d5e7f9a1b2c4d6e8f0a2b4c6d8e0f1a3b5c7d9e1f2a4b6c",
       "requested_by": "exam_scheduling_agent"}'
```
""".strip()

TAGS: list[dict[str, Any]] = [
    {"name": "appointments", "description": "Scheduling requests (authenticated)."},
    {"name": "exams", "description": "Catalog lookup (authenticated)."},
    {"name": "health", "description": "Liveness and readiness probes (public)."},
]

PATIENT_REF = "pt_K7Q2M4XZB3C6D7E5F6G7H2J4KA"
DOCUMENT_SHA256 = "9b1e2c4d6f8a0b1c3d5e7f9a1b2c4d6e8f0a2b4c6d8e0f1a3b5c7d9e1f2a4b6c"
APPOINTMENT_ID = "apt_01JB8Z3K4N5P6Q7R8S9T0V1W2X"
REQUEST_ID = "6f1d2c3b-4a59-4e8f-9d7c-1b2a3c4d5e6f"
REQUESTED_BY = "exam_scheduling_agent"
UNKNOWN_EXAM_CODE = "LAB-99999"
COMPLETE_BLOOD_COUNT = {"code": "LAB-10001", "name": "Hemograma completo"}
FASTING_GLUCOSE = {"code": "LAB-10012", "name": "Glicemia de jejum"}

APPOINTMENT_EXAMPLE: dict[str, Any] = {
    "appointment_id": APPOINTMENT_ID,
    "status": "requested",
    "patient_ref": PATIENT_REF,
    "exams": [COMPLETE_BLOOD_COUNT, FASTING_GLUCOSE],
    "source_document_sha256": DOCUMENT_SHA256,
    "requested_by": REQUESTED_BY,
    "created_at": "2026-10-26T22:48:00Z",
}

CREATE_EXAMPLES: dict[str, Any] = {
    "two_exams": {
        "summary": "Two catalog exams",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [COMPLETE_BLOOD_COUNT, FASTING_GLUCOSE],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": REQUESTED_BY,
        },
    },
    "unknown_code": {
        "summary": "Unknown code (422)",
        "description": f"{UNKNOWN_EXAM_CODE} is not in the catalog, so the request is rejected.",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [{"code": UNKNOWN_EXAM_CODE, "name": "Exame inexistente"}],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": REQUESTED_BY,
        },
    },
    "personal_data": {
        "summary": "Personal data field (422)",
        "description": "Fields outside the contract, such as a patient name, are rejected.",
        "value": {
            "patient_ref": PATIENT_REF,
            "patient_name": "Fulano de Tal",
            "exams": [COMPLETE_BLOOD_COUNT],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": REQUESTED_BY,
        },
    },
}

IDEMPOTENCY_KEY_EXAMPLE = "3f0c9a8e5b7d4c21a6e9f0b1c2d3e4f5"

REQUEST_ID_HEADER_DOC = {
    "description": "Request id: the `X-Request-ID` sent by the client, or a generated UUID.",
    "schema": {"type": "string"},
}

_CREATE_PATH = "/v1/appointments"
_APPOINTMENT_PATH = f"/v1/appointments/{APPOINTMENT_ID}"
_UNKNOWN_EXAM_PATH = f"/v1/exams/{UNKNOWN_EXAM_CODE}"

_DESCRIPTION_BY_STATUS = {
    400: "`Idempotency-Key` missing or invalid.",
    401: "API key missing or invalid.",
    404: "Not found.",
    409: "Same `Idempotency-Key` used with a different body.",
    422: "Request does not satisfy the contract.",
    500: "Unexpected error; details are only in the server log.",
}


def _problem_example(  # noqa: PLR0913, PLR0917 - one argument per problem field
    code: str,
    title: str,
    status: int,
    detail: str | None,
    instance: str,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    problem = Problem(
        type=problem_type(code),
        title=title,
        status=status,
        detail=detail,
        instance=instance,
        request_id=REQUEST_ID,
        errors=errors,  # type: ignore[arg-type]  # validated into FieldError
    )
    return problem.model_dump(mode="json", exclude_none=True)


PROBLEM_EXAMPLES: dict[str, dict[str, Any]] = {
    "invalid_idempotency_key": _problem_example(
        "invalid_idempotency_key",
        PROBLEM_TITLES["invalid_idempotency_key"],
        400,
        INVALID_IDEMPOTENCY_KEY_DETAIL,
        _CREATE_PATH,
        [{"loc": ["header", "Idempotency-Key"], "code": "missing", "msg": "Field required"}],
    ),
    "unauthorized": _problem_example(
        "unauthorized",
        "Unauthorized",
        401,
        INVALID_API_KEY_DETAIL,
        _CREATE_PATH,
    ),
    "idempotency_key_reuse": _problem_example(
        "idempotency_key_reuse",
        PROBLEM_TITLES["idempotency_key_reuse"],
        409,
        IDEMPOTENCY_KEY_REUSE_DETAIL,
        _CREATE_PATH,
    ),
    "unknown_exam_code": _problem_example(
        "unknown_exam_code",
        PROBLEM_TITLES["unknown_exam_code"],
        422,
        "1 exam code(s) do not exist in the catalog.",
        _CREATE_PATH,
        [
            {
                "loc": ["body", "exams", 1, "code"],
                "code": "unknown_exam_code",
                "msg": f"{UNKNOWN_EXAM_CODE} not found",
            }
        ],
    ),
    "exam_name_mismatch": _problem_example(
        "exam_name_mismatch",
        PROBLEM_TITLES["exam_name_mismatch"],
        422,
        "1 exam name(s) do not match the catalog.",
        _CREATE_PATH,
        [
            {
                "loc": ["body", "exams", 0, "name"],
                "code": "exam_name_mismatch",
                "msg": "name does not match the catalog name for LAB-10001: 'Hemograma completo'",
            }
        ],
    ),
    "validation_error": _problem_example(
        "validation_error",
        PROBLEM_TITLES["validation_error"],
        422,
        "1 validation error(s).",
        _CREATE_PATH,
        [
            {
                "loc": ["body", "patient_name"],
                "code": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
            }
        ],
    ),
    "appointment_not_found": _problem_example(
        "appointment_not_found",
        PROBLEM_TITLES["appointment_not_found"],
        404,
        f"{APPOINTMENT_ID} does not exist.",
        _APPOINTMENT_PATH,
    ),
    "exam_not_found": _problem_example(
        "exam_not_found",
        PROBLEM_TITLES["exam_not_found"],
        404,
        f"{UNKNOWN_EXAM_CODE} is not in the catalog.",
        _UNKNOWN_EXAM_PATH,
    ),
    "internal_error": _problem_example(
        "internal_error",
        PROBLEM_TITLES["internal_error"],
        500,
        INTERNAL_ERROR_DETAIL,
        _CREATE_PATH,
    ),
}


def problem_responses(
    codes_by_status: Mapping[int, Sequence[str]],
) -> dict[int | str, dict[str, Any]]:
    responses: dict[int | str, dict[str, Any]] = {
        status: _problem_response_doc(status, codes) for status, codes in codes_by_status.items()
    }
    return responses


def fix_problem_media_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Moves the `Problem` schema FastAPI put under application/json to problem+json."""
    for response in _responses(schema):
        content = response.get("content", {})
        if PROBLEM_MEDIA_TYPE in content and "application/json" in content:
            content[PROBLEM_MEDIA_TYPE]["schema"] = content.pop("application/json")["schema"]

    return schema


def _problem_response_doc(status: int, codes: Sequence[str]) -> dict[str, Any]:
    examples = {
        code: {"summary": PROBLEM_EXAMPLES[code]["title"], "value": PROBLEM_EXAMPLES[code]}
        for code in codes
    }
    return {
        "model": Problem,
        "description": _DESCRIPTION_BY_STATUS[status],
        "headers": {REQUEST_ID_HEADER: REQUEST_ID_HEADER_DOC},
        "content": {PROBLEM_MEDIA_TYPE: {"examples": examples}},
    }


def _responses(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            yield from operation.get("responses", {}).values()
