"""MCP server (SSE transport) with search over the exam catalog.

Tools: `batch_search` (the one agent specs allow), `search_exams` and
`get_exam` (operation and debugging); resource `catalog://info`. Results only
rank and score catalog exams; accepting a candidate is the runtime's job.

Every path except `/health` requires `Authorization: Bearer <MCP_AUTH_TOKEN>`;
the check wraps the whole SSE app, so both `/sse` and `/messages/` are covered.

Errors are tool results with `isError: true` and a JSON body
`{code, message, retryable}` (`RagToolError`), built here instead of raised so
FastMCP does not prefix the message. Arguments are checked here as well, so a
range error has the same shape as any other error. Logs carry counts and
timings only, never the mentions, under the caller's `request_id` (the agent
sends its run id) when it is a UUID.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from lab_common import logging as lab_logging
from lab_common.contracts.rag import (
    EXAM_CODE_PATTERN,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    BatchSearchResult,
    Exam,
    RagErrorCode,
    RagToolError,
    SearchResult,
)
from lab_common.mcp_server import BearerAuthMiddleware, logged_request_id
from rag_mcp.catalog import Catalog
from rag_mcp.logs import SERVICE, configure_logging
from rag_mcp.scoring import ExamSearcher
from rag_mcp.settings import RagSettings

DEFAULT_TOP_K = 5
CATALOG_INFO_URI = "catalog://info"
HEALTH_PATH = "/health"
_CODE = re.compile(EXAM_CODE_PATTERN)

SEARCH_DESCRIPTION = (
    "Rank catalog exams for one exam mention (as written in the order). Each candidate "
    "has a score in [0, 1], comparable across queries; 1.0 is an exact name or alias. "
    "The tool does not decide which candidate is right."
)
BATCH_DESCRIPTION = (
    "Rank catalog exams for up to {max_batch} exam mentions in one call; results come "
    "in the order of the queries. Same scores as search_exams."
)
GET_EXAM_DESCRIPTION = "Return the catalog entry for an exam code (LAB-NNNNN)."

log = lab_logging.get_logger(__name__)


class RagError(Exception):
    def __init__(self, code: RagErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return RagToolError(code=self.code, message=self.message).model_dump(mode="json")


def _invalid(message: str) -> RagError:
    return RagError(RagErrorCode.INVALID_ARGUMENT, message)


def _check_top_k(top_k: object) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise _invalid(f"top_k must be an integer from 1 to {MAX_TOP_K}")
    return top_k


def _check_query(query: object, name: str = "query") -> str:
    if not isinstance(query, str) or not 1 <= len(query) <= MAX_QUERY_CHARS:
        raise _invalid(f"{name} must be a string of 1 to {MAX_QUERY_CHARS} characters")
    return query


class RagService:
    def __init__(self, catalog: Catalog, max_batch: int) -> None:
        self._searcher = ExamSearcher(catalog)
        self._max_batch = max_batch

    @property
    def catalog(self) -> Catalog:
        return self._searcher.catalog

    @property
    def max_batch(self) -> int:
        return self._max_batch

    def search_exams(self, query: object, top_k: object) -> SearchResult:
        return self._searcher.search(_check_query(query), _check_top_k(top_k))

    def batch_search(self, queries: object, top_k: object) -> BatchSearchResult:
        if not isinstance(queries, list) or not 1 <= len(queries) <= self._max_batch:
            raise _invalid(f"queries must be a list of 1 to {self._max_batch} strings")
        checked = [_check_query(q, f"queries[{i}]") for i, q in enumerate(queries)]
        k = _check_top_k(top_k)
        catalog = self.catalog
        return BatchSearchResult(
            catalog_version=catalog.version,
            catalog_sha256=catalog.sha256,
            results=[self._searcher.search(query, k) for query in checked],
        )

    def get_exam(self, code: object) -> Exam:
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise _invalid("code must look like LAB-NNNNN")
        exam = self.catalog.get(code)
        if exam is None:
            raise RagError(RagErrorCode.NOT_FOUND, "no exam with this code in the catalog")
        return exam


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _json(data: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structuredContent=data,
        isError=is_error,
    )


def _run[R: BaseModel](
    tool: str,
    call: Callable[[], R],
    summarize: Callable[[R], dict[str, int]],
    request_id: object = None,
) -> CallToolResult:
    started = time.perf_counter()
    with structlog.contextvars.bound_contextvars(request_id=logged_request_id(request_id)):
        try:
            result = call()
        except RagError as error:
            log.warning(
                "rag_rejected", tool=tool, error_code=error.code.value, duration_ms=_ms(started)
            )
            return _json(error.as_dict(), is_error=True)
        log.info("rag_done", tool=tool, duration_ms=_ms(started), **summarize(result))
    return _json(result.model_dump(mode="json"))


def _search_counts(result: SearchResult) -> dict[str, int]:
    return {"queries": 1, "empty_results": int(not result.candidates)}


def _batch_counts(result: BatchSearchResult) -> dict[str, int]:
    return {
        "queries": len(result.results),
        "empty_results": sum(1 for r in result.results if not r.candidates),
    }


def _no_counts(result: Exam) -> dict[str, int]:
    return {}


def _ensure_json_logging(settings: RagSettings) -> None:
    # FastMCP calls logging.basicConfig with a Rich handler, which would print
    # records without the redaction processor; it is a no-op once the root
    # logger has a handler.
    if not logging.getLogger().handlers:
        configure_logging(settings.log_level)


def build_server(service: RagService, settings: RagSettings) -> FastMCP:
    _ensure_json_logging(settings)
    mcp = FastMCP(SERVICE, host=settings.host, port=settings.port, log_level=settings.log_level)
    info = service.catalog.info().model_dump(mode="json")

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", **info})

    @mcp.resource(CATALOG_INFO_URI, name="catalog_info", mime_type="application/json")
    def catalog_info() -> str:
        return json.dumps(info)

    @mcp.tool(name="search_exams", description=SEARCH_DESCRIPTION, structured_output=False)
    def search_exams(
        query: str, top_k: int = DEFAULT_TOP_K, request_id: str | None = None
    ) -> CallToolResult:
        return _run(
            "search_exams",
            lambda: service.search_exams(query, top_k),
            _search_counts,
            request_id,
        )

    @mcp.tool(
        name="batch_search",
        description=BATCH_DESCRIPTION.format(max_batch=service.max_batch),
        structured_output=False,
    )
    def batch_search(
        queries: list[str], top_k: int = DEFAULT_TOP_K, request_id: str | None = None
    ) -> CallToolResult:
        return _run(
            "batch_search",
            lambda: service.batch_search(queries, top_k),
            _batch_counts,
            request_id,
        )

    @mcp.tool(name="get_exam", description=GET_EXAM_DESCRIPTION, structured_output=False)
    def get_exam(code: str) -> CallToolResult:
        return _run("get_exam", lambda: service.get_exam(code), _no_counts)

    return mcp


def build_app(settings: RagSettings, catalog: Catalog) -> ASGIApp:
    service = RagService(catalog, settings.max_batch)
    return BearerAuthMiddleware(
        build_server(service, settings).sse_app(),
        token=settings.mcp_auth_token.get_secret_value(),
        public_paths=frozenset({HEALTH_PATH}),
    )
