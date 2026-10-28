from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

CONTRACT_VERSION: Literal["1"] = "1"
EXAM_CODE_PATTERN = r"^LAB-[0-9]{5}$"
CATALOG_VERSION_PATTERN = r"^[0-9]{4}\.[0-9]{2}\.[0-9]+$"
MAX_QUERY_CHARS = 200
MAX_TOP_K = 10
MAX_BATCH_QUERIES = 50

ExamCode = Annotated[str, Field(pattern=EXAM_CODE_PATTERN)]
CatalogVersion = Annotated[str, Field(pattern=CATALOG_VERSION_PATTERN)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[float, Field(ge=0, le=1)]
NonEmpty = Annotated[str, Field(min_length=1)]

MatchedOn = Literal["exact_name", "exact_alias", "name", "alias"]
"""`exact_*`: the normalized query equals that field (score 1.0); otherwise the best fuzzy match."""


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Exam(_Contract):
    code: ExamCode
    name: NonEmpty
    aliases: list[NonEmpty]
    category: NonEmpty
    specimen: NonEmpty
    preparation: NonEmpty


class Candidate(_Contract):
    code: ExamCode
    name: str
    score: Score
    matched_on: MatchedOn
    category: str


# The RAG only ranks candidates; accepting one is the runtime confidence gate's decision.
class SearchResult(_Contract):
    contract_version: Literal["1"] = CONTRACT_VERSION
    catalog_version: CatalogVersion
    catalog_sha256: Sha256
    query: str
    normalized_query: str
    """Empty when nothing searchable was left."""
    candidates: list[Candidate]
    """Ordered by (-score, code); at most `top_k` items."""


class BatchSearchResult(_Contract):
    contract_version: Literal["1"] = CONTRACT_VERSION
    catalog_version: CatalogVersion
    catalog_sha256: Sha256
    results: list[SearchResult]
    """One result per query, in the order the queries were sent."""


class CatalogInfo(_Contract):
    catalog_version: CatalogVersion
    catalog_sha256: Sha256
    exam_count: NonNegativeInt


class RagErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"


class RagToolError(_Contract):
    """Body of a tool error result (`isError: true`). No RAG error is retryable:
    the index lives in memory, so a repeated call gives the same answer."""

    code: RagErrorCode
    message: str
    retryable: Literal[False] = False
