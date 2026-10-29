from typing import Annotated

from fastapi import APIRouter, Depends, Path, status

from scheduler_api.api import docs
from scheduler_api.api.deps import CatalogDep, document_request_id, require_api_key
from scheduler_api.api.errors import PROBLEM_TITLES, ApiError
from scheduler_api.api.middleware import REQUEST_ID_HEADER
from scheduler_api.schemas import EXAM_CODE_PATTERN, CatalogEntry

router = APIRouter(
    prefix="/v1",
    tags=[docs.TAG_EXAMS],
    dependencies=[Depends(require_api_key), Depends(document_request_id)],
)


@router.get(
    "/exams/{code}",
    summary="Consultar um exame do catálogo",
    description="Devolve o nome oficial de um código, o mesmo usado na validação do `POST`.",
    response_description="O exame do catálogo.",
    responses={
        200: {
            "content": {"application/json": {"example": docs.COMPLETE_BLOOD_COUNT}},
            "headers": {REQUEST_ID_HEADER: docs.REQUEST_ID_HEADER_DOC},
        },
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                404: ["exam_not_found"],
                422: ["invalid_exam_code"],
                500: ["internal_error"],
            }
        ),
    },
)
def get_exam(
    code: Annotated[
        str,
        Path(
            pattern=EXAM_CODE_PATTERN,
            description="Código do catálogo (três letras, hífen e cinco dígitos).",
            openapi_examples={
                "hemograma": {
                    "summary": "Hemograma completo",
                    "value": docs.COMPLETE_BLOOD_COUNT["code"],
                },
                "glicemia": {"summary": "Glicemia de jejum", "value": docs.FASTING_GLUCOSE["code"]},
            },
        ),
    ],
    catalog: CatalogDep,
) -> CatalogEntry:
    exam = catalog.get(code)
    if exam is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "exam_not_found",
            PROBLEM_TITLES["exam_not_found"],
            f"{code} is not in the catalog.",
        )

    return CatalogEntry(code=exam.code, name=exam.name)
