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

TITLE = "API de Agendamento de Exames"

DESCRIPTION = """
Registra e consulta **solicitações de agendamento de exames** enviadas pelo agente.
Cada solicitação fica com status `requested`; reservar horário está fora do escopo.

**Autenticação:** as rotas `/v1` exigem o header `X-API-Key`. Clique em **Authorize**
e informe a chave configurada no servidor (`SCHEDULER_API_KEY`).

**Sem dados pessoais:** o paciente é identificado só pelo pseudônimo `patient_ref`
(`pt_` + 26 caracteres). Campos fora do contrato, como nome ou CPF, são rejeitados com `422`.

Os erros usam `application/problem+json` (RFC 9457) e trazem o `request_id`, o mesmo
valor do header `X-Request-ID` da resposta.
""".strip()

TAG_APPOINTMENTS = "Agendamentos"
TAG_EXAMS = "Exames"
TAG_HEALTH = "Health"

TAGS: list[dict[str, Any]] = [
    {
        "name": TAG_APPOINTMENTS,
        "description": "Registro e consulta de solicitações de agendamento (exige X-API-Key).",
    },
    {"name": TAG_EXAMS, "description": "Consulta ao catálogo de exames (exige X-API-Key)."},
    {"name": TAG_HEALTH, "description": "Sondas de liveness e readiness (públicas)."},
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
        "summary": "Dois exames do catálogo",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [COMPLETE_BLOOD_COUNT, FASTING_GLUCOSE],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": REQUESTED_BY,
        },
    },
    "unknown_code": {
        "summary": "Código fora do catálogo (422)",
        "description": f"{UNKNOWN_EXAM_CODE} não existe no catálogo; a solicitação é rejeitada.",
        "value": {
            "patient_ref": PATIENT_REF,
            "exams": [{"code": UNKNOWN_EXAM_CODE, "name": "Exame inexistente"}],
            "source_document_sha256": DOCUMENT_SHA256,
            "requested_by": REQUESTED_BY,
        },
    },
    "personal_data": {
        "summary": "Campo com dado pessoal (422)",
        "description": "Campos fora do contrato, como o nome do paciente, são rejeitados.",
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
    "description": "Id da requisição: o `X-Request-ID` enviado pelo cliente ou um UUID gerado.",
    "schema": {"type": "string", "maxLength": 128},
    "example": REQUEST_ID,
}

_APPOINTMENTS_PATH = "/v1/appointments"
_APPOINTMENT_PATH = f"/v1/appointments/{APPOINTMENT_ID}"
_UNKNOWN_EXAM_PATH = f"/v1/exams/{UNKNOWN_EXAM_CODE}"

_DESCRIPTION_BY_STATUS = {
    400: "`Idempotency-Key` ausente ou inválido.",
    401: "API key ausente ou inválida.",
    404: "Não encontrado.",
    409: "Mesmo `Idempotency-Key` usado com outro corpo.",
    422: "A requisição não atende ao contrato.",
    500: "Erro inesperado; os detalhes ficam só no log do servidor.",
}


def _problem_example(  # noqa: PLR0913, PLR0917 - one argument per problem field
    summary: str,
    code: str,
    status: int,
    detail: str | None,
    instance: str,
    errors: list[dict[str, Any]] | None = None,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    problem = Problem(
        type=problem_type(code),
        title=title or PROBLEM_TITLES[code],
        status=status,
        detail=detail,
        instance=instance,
        request_id=REQUEST_ID,
        errors=errors,  # type: ignore[arg-type]  # validated into FieldError
    )
    return {"summary": summary, "value": problem.model_dump(mode="json", exclude_none=True)}


def _validation_example(summary: str, instance: str, error: dict[str, Any]) -> dict[str, Any]:
    return _problem_example(
        summary, "validation_error", 422, "1 validation error(s).", instance, [error]
    )


PROBLEM_EXAMPLES: dict[str, dict[str, Any]] = {
    "invalid_idempotency_key": _problem_example(
        "Idempotency-Key ausente",
        "invalid_idempotency_key",
        400,
        INVALID_IDEMPOTENCY_KEY_DETAIL,
        _APPOINTMENTS_PATH,
        [{"loc": ["header", "Idempotency-Key"], "code": "missing", "msg": "Field required"}],
    ),
    "unauthorized": _problem_example(
        "Sem API key ou com a chave errada",
        "unauthorized",
        401,
        INVALID_API_KEY_DETAIL,
        _APPOINTMENTS_PATH,
        title="Unauthorized",
    ),
    "idempotency_key_reuse": _problem_example(
        "Mesma chave com outro corpo",
        "idempotency_key_reuse",
        409,
        IDEMPOTENCY_KEY_REUSE_DETAIL,
        _APPOINTMENTS_PATH,
    ),
    "unknown_exam_code": _problem_example(
        "Código fora do catálogo",
        "unknown_exam_code",
        422,
        "1 exam code(s) do not exist in the catalog.",
        _APPOINTMENTS_PATH,
        [
            {
                "loc": ["body", "exams", 1, "code"],
                "code": "unknown_exam_code",
                "msg": f"{UNKNOWN_EXAM_CODE} not found",
            }
        ],
    ),
    "exam_name_mismatch": _problem_example(
        "Nome diferente do catálogo",
        "exam_name_mismatch",
        422,
        "1 exam name(s) do not match the catalog.",
        _APPOINTMENTS_PATH,
        [
            {
                "loc": ["body", "exams", 0, "name"],
                "code": "exam_name_mismatch",
                "msg": "name does not match the catalog name for LAB-10001: 'Hemograma completo'",
            }
        ],
    ),
    "validation_error": _validation_example(
        "Campo fora do contrato",
        _APPOINTMENTS_PATH,
        {
            "loc": ["body", "patient_name"],
            "code": "extra_forbidden",
            "msg": "Extra inputs are not permitted",
        },
    ),
    "invalid_limit": _validation_example(
        "limit fora de 1 a 100",
        _APPOINTMENTS_PATH,
        {
            "loc": ["query", "limit"],
            "code": "less_than_equal",
            "msg": "Input should be less than or equal to 100",
        },
    ),
    "invalid_patient_ref": _validation_example(
        "patient_ref fora do formato",
        _APPOINTMENTS_PATH,
        {
            "loc": ["query", "patient_ref"],
            "code": "string_pattern_mismatch",
            "msg": "String should match pattern '^pt_[A-Z2-7]{26}$'",
        },
    ),
    "invalid_appointment_id": _validation_example(
        "appointment_id fora do formato",
        "/v1/appointments/123",
        {
            "loc": ["path", "appointment_id"],
            "code": "string_pattern_mismatch",
            "msg": "String should match pattern '^apt_[0-9A-HJKMNP-TV-Z]{26}$'",
        },
    ),
    "invalid_exam_code": _validation_example(
        "Código fora do formato",
        "/v1/exams/lab-1",
        {
            "loc": ["path", "code"],
            "code": "string_pattern_mismatch",
            "msg": "String should match pattern '^[A-Z]{3}-\\d{5}$'",
        },
    ),
    "appointment_not_found": _problem_example(
        "Solicitação inexistente",
        "appointment_not_found",
        404,
        f"{APPOINTMENT_ID} does not exist.",
        _APPOINTMENT_PATH,
    ),
    "exam_not_found": _problem_example(
        "Código fora do catálogo",
        "exam_not_found",
        404,
        f"{UNKNOWN_EXAM_CODE} is not in the catalog.",
        _UNKNOWN_EXAM_PATH,
    ),
    "internal_error": _problem_example(
        "Erro inesperado",
        "internal_error",
        500,
        INTERNAL_ERROR_DETAIL,
        _APPOINTMENTS_PATH,
    ),
}


def problem_responses(
    examples_by_status: Mapping[int, Sequence[str]],
) -> dict[int | str, dict[str, Any]]:
    responses: dict[int | str, dict[str, Any]] = {
        status: _problem_response_doc(status, names) for status, names in examples_by_status.items()
    }
    return responses


def fix_problem_media_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Moves the `Problem` schema FastAPI put under application/json to problem+json."""
    for response in _responses(schema):
        content = response.get("content", {})
        if PROBLEM_MEDIA_TYPE in content and "application/json" in content:
            content[PROBLEM_MEDIA_TYPE]["schema"] = content.pop("application/json")["schema"]

    return schema


def _problem_response_doc(status: int, names: Sequence[str]) -> dict[str, Any]:
    return {
        "model": Problem,
        "description": _DESCRIPTION_BY_STATUS[status],
        "headers": {REQUEST_ID_HEADER: REQUEST_ID_HEADER_DOC},
        "content": {PROBLEM_MEDIA_TYPE: {"examples": {n: PROBLEM_EXAMPLES[n] for n in names}}},
    }


def _responses(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            yield from operation.get("responses", {}).values()
