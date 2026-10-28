from typing import Annotated

from fastapi import APIRouter, Depends, Path, status

from scheduler_api.api import docs
from scheduler_api.api.deps import CatalogDep, document_request_id, require_api_key
from scheduler_api.api.errors import PROBLEM_TITLES, ApiError
from scheduler_api.schemas import EXAM_CODE_PATTERN, ExamOut

router = APIRouter(
    prefix="/v1",
    tags=["exams"],
    dependencies=[Depends(require_api_key), Depends(document_request_id)],
)


@router.get(
    "/exams/{code}",
    summary="Get a catalog exam",
    description="Returns the official name of a catalog code, as the API validates it.",
    response_description="The catalog entry.",
    responses={
        200: {"content": {"application/json": {"example": docs.COMPLETE_BLOOD_COUNT}}},
        **docs.problem_responses(
            {
                401: ["unauthorized"],
                404: ["exam_not_found"],
                422: ["validation_error"],
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
            description="Catalog code.",
            examples=[docs.COMPLETE_BLOOD_COUNT["code"]],
        ),
    ],
    catalog: CatalogDep,
) -> ExamOut:
    exam = catalog.get(code)
    if exam is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "exam_not_found",
            PROBLEM_TITLES["exam_not_found"],
            f"{code} is not in the catalog.",
        )

    return ExamOut(code=exam.code, name=exam.name)
