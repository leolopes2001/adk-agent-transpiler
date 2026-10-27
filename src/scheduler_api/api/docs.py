"""OpenAPI documentation: description, tags, examples and documented responses.

The routes declare their responses with the helpers below so Swagger shows
exactly what the API answers, including every problem type with an example.
`fix_problem_media_types` then moves error schemas to
`application/problem+json`, the media type the handlers actually send.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from scheduler_api.api.errors import PROBLEM_MEDIA_TYPE, Problem, problem_type

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

APPOINTMENT_EXAMPLE: dict[str, Any] = {
    "appointment_id": APPOINTMENT_ID,
    "status": "requested",
    "patient_ref": PATIENT_REF,
    "exams": [
        {"code": "LAB-10001", "name": "Hemograma completo"},
        {"code": "LAB-10012", "name": "Glicemia de jejum"},
    ],
    "source_document_sha256": DOCUMENT_SHA256,
    "requested_by": "exam_scheduling_agent",
    "created_at": "2026-10-26T22:48:00Z",
}

CREATE_EXAMPLES: dict[str, Any] = {
    "two_exams": {
        "summary": "Two catalog exams",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [
                {"code": "LAB-10001", "name": "Hemograma completo"},
                {"code": "LAB-10012", "name": "Glicemia de jejum"},
            ],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": "exam_scheduling_agent",
        },
    },
    "unknown_code": {
        "summary": "Unknown code (422)",
        "description": "LAB-99999 is not in the catalog, so the request is rejected.",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [{"code": "LAB-99999", "name": "Exame inexistente"}],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": "exam_scheduling_agent",
        },
    },
    "personal_data": {
        "summary": "Personal data field (422)",
        "description": "Fields outside the contract, such as a patient name, are rejected.",
        "value": {
            "patient_ref": PATIENT_REF,
            "patient_name": "Fulano de Tal",
            "exams": [{"code": "LAB-10001", "name": "Hemograma completo"}],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": "exam_scheduling_agent",
        },
    },
}

IDEMPOTENCY_KEY_EXAMPLE = "3f0c9a8e5b7d4c21a6e9f0b1c2d3e4f5"


def _problem(  # noqa: PLR0913, PLR0917 - one argument per problem field
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


_POST = "/v1/appointments"
_GET = f"/v1/appointments/{APPOINTMENT_ID}"
_EXAM = "/v1/exams/LAB-99999"

PROBLEM_EXAMPLES: dict[str, dict[str, Any]] = {
    "invalid_idempotency_key": _problem(
        "invalid_idempotency_key",
        "Invalid Idempotency-Key",
        400,
        "Idempotency-Key is required: 8 to 128 characters from A-Z a-z 0-9 _ - : .",
        _POST,
        [{"loc": ["header", "Idempotency-Key"], "code": "missing", "msg": "Field required"}],
    ),
    "unauthorized": _problem(
        "unauthorized", "Unauthorized", 401, "Missing or invalid API key.", _POST
    ),
    "idempotency_key_reuse": _problem(
        "idempotency_key_reuse",
        "Idempotency key reuse",
        409,
        "This Idempotency-Key was already used with a different request body.",
        _POST,
    ),
    "unknown_exam_code": _problem(
        "unknown_exam_code",
        "Unknown exam code",
        422,
        "1 exam code(s) do not exist in the catalog.",
        _POST,
        [
            {
                "loc": ["body", "exams", 1, "code"],
                "code": "unknown_exam_code",
                "msg": "LAB-99999 not found",
            }
        ],
    ),
    "exam_name_mismatch": _problem(
        "exam_name_mismatch",
        "Exam name mismatch",
        422,
        "1 exam name(s) do not match the catalog.",
        _POST,
        [
            {
                "loc": ["body", "exams", 0, "name"],
                "code": "exam_name_mismatch",
                "msg": "name does not match the catalog name for LAB-10001: 'Hemograma completo'",
            }
        ],
    ),
    "validation_error": _problem(
        "validation_error",
        "Request validation failed",
        422,
        "1 validation error(s).",
        _POST,
        [
            {
                "loc": ["body", "patient_name"],
                "code": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
            }
        ],
    ),
    "appointment_not_found": _problem(
        "appointment_not_found",
        "Appointment not found",
        404,
        f"{APPOINTMENT_ID} does not exist.",
        _GET,
    ),
    "exam_not_found": _problem(
        "exam_not_found", "Exam not found", 404, "LAB-99999 is not in the catalog.", _EXAM
    ),
    "internal_error": _problem(
        "internal_error",
        "Internal server error",
        500,
        "The request could not be completed. Quote the request_id when reporting it.",
        _POST,
    ),
}

_DESCRIPTIONS = {
    400: "`Idempotency-Key` missing or invalid.",
    401: "API key missing or invalid.",
    404: "Not found.",
    409: "Same `Idempotency-Key` used with a different body.",
    422: "Request does not satisfy the contract.",
    500: "Unexpected error; details are only in the server log.",
}

REQUEST_ID_HEADER_DOC = {
    "description": "Request id: the `X-Request-ID` sent by the client, or a generated UUID.",
    "schema": {"type": "string"},
}


def problem_responses(
    codes_by_status: Mapping[int, Sequence[str]],
) -> dict[int | str, dict[str, Any]]:
    """`responses=` entries documenting the given problem codes, keyed by status."""
    responses: dict[int | str, dict[str, Any]] = {}
    for status, codes in codes_by_status.items():
        responses[status] = {
            "model": Problem,
            "description": _DESCRIPTIONS[status],
            "headers": {"X-Request-ID": REQUEST_ID_HEADER_DOC},
            "content": {
                PROBLEM_MEDIA_TYPE: {
                    "examples": {
                        code: {
                            "summary": PROBLEM_EXAMPLES[code]["title"],
                            "value": PROBLEM_EXAMPLES[code],
                        }
                        for code in codes
                    }
                }
            },
        }
    return responses


def fix_problem_media_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Moves the `Problem` schema FastAPI put under application/json to problem+json."""
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            for response in operation.get("responses", {}).values():
                content = response.get("content", {})
                if PROBLEM_MEDIA_TYPE in content and "application/json" in content:
                    content[PROBLEM_MEDIA_TYPE]["schema"] = content.pop("application/json")[
                        "schema"
                    ]
    return schema
