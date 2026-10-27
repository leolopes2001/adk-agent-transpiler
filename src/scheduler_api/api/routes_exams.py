"""/v1/exams: catalog lookup, for operation and debugging."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, status

from scheduler_api.api.deps import CatalogDep, require_api_key
from scheduler_api.api.errors import ApiError
from scheduler_api.schemas import EXAM_CODE_PATTERN, ExamOut

router = APIRouter(prefix="/v1", tags=["exams"], dependencies=[Depends(require_api_key)])


@router.get("/exams/{code}", summary="Get a catalog exam")
def get_exam(
    code: Annotated[str, Path(pattern=EXAM_CODE_PATTERN)],
    catalog: CatalogDep,
) -> ExamOut:
    exam = catalog.get(code)
    if exam is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "exam_not_found",
            "Exam not found",
            f"{code} is not in the catalog.",
        )
    return ExamOut(code=exam.code, name=exam.name)
