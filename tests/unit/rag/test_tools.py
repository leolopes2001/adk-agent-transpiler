import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from starlette.testclient import TestClient

from lab_common import logging as lab_logging
from lab_common.contracts.rag import BatchSearchResult, CatalogInfo, Exam, SearchResult
from rag_mcp import __main__ as rag_main
from rag_mcp.catalog import Catalog, load_catalog
from rag_mcp.logs import REDACTED, LibraryLogFilter, configure_logging
from rag_mcp.server import RagService, build_app, build_server
from rag_mcp.settings import RagSettings

pytestmark = pytest.mark.unit

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "exams_catalog.json"
TOKEN = "unit-token-" + "x" * 32
MENTIONS = ["1. Hemograma completo", "Glicemla de jejum", "T4", "Ressonância de crânio"]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(CATALOG_PATH)


@pytest.fixture(autouse=True)
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    configure_logging(stream=stream)
    yield stream
    if lab_logging._handler is not None:
        logging.getLogger().removeHandler(lab_logging._handler)
        lab_logging._handler = None
    structlog.reset_defaults()


def settings(**overrides: Any) -> RagSettings:
    return RagSettings(mcp_auth_token=TOKEN, **overrides)


@pytest.fixture
def mcp(catalog: Catalog) -> FastMCP:
    config = settings()
    return build_server(RagService(catalog, config.max_batch), config)


async def call(mcp: FastMCP, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    result = await mcp.call_tool(tool, arguments)
    assert isinstance(result, CallToolResult)
    return result


def body(result: CallToolResult) -> dict[str, Any]:
    [content] = result.content
    assert isinstance(content, TextContent)
    parsed: dict[str, Any] = json.loads(content.text)
    assert parsed == result.structuredContent
    return parsed


def ok(result: CallToolResult) -> dict[str, Any]:
    assert not result.isError, body(result)
    return body(result)


def error(result: CallToolResult) -> dict[str, Any]:
    assert result.isError
    return body(result)


def invalid(message: str) -> dict[str, Any]:
    return {"code": "INVALID_ARGUMENT", "message": message, "retryable": False}


class TestListing:
    async def test_exposes_three_tools(self, mcp: FastMCP) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}

        assert sorted(tools) == ["batch_search", "get_exam", "search_exams"]
        assert tools["batch_search"].inputSchema["required"] == ["queries"]
        assert tools["batch_search"].inputSchema["properties"]["top_k"]["default"] == 5
        assert tools["search_exams"].inputSchema["required"] == ["query"]
        assert tools["get_exam"].inputSchema["required"] == ["code"]
        assert "50" in (tools["batch_search"].description or "")

    async def test_catalog_info_resource(self, mcp: FastMCP, catalog: Catalog) -> None:
        [content] = await mcp.read_resource("catalog://info")

        info = CatalogInfo.model_validate_json(content.content)
        assert info == catalog.info()
        assert content.mime_type == "application/json"


class TestSearchExams:
    async def test_returns_a_search_result(self, mcp: FastMCP, catalog: Catalog) -> None:
        result = SearchResult.model_validate(
            ok(await call(mcp, "search_exams", {"query": "7. PCR", "top_k": 2}))
        )

        assert [c.code for c in result.candidates] == ["LAB-10037", "LAB-10104"]
        assert result.candidates[0].matched_on == "exact_alias"
        assert (result.catalog_version, result.catalog_sha256) == (catalog.version, catalog.sha256)

    async def test_default_top_k_is_five(self, mcp: FastMCP) -> None:
        result = ok(await call(mcp, "search_exams", {"query": "colesterol"}))
        assert len(result["candidates"]) == 5

    async def test_nothing_searchable_is_an_empty_result(self, mcp: FastMCP) -> None:
        result = ok(await call(mcp, "search_exams", {"query": " ;; "}))
        assert (result["normalized_query"], result["candidates"]) == ("", [])

    @pytest.mark.parametrize("query", ["", "x" * 201])
    async def test_query_length_is_checked(self, mcp: FastMCP, query: str) -> None:
        assert error(await call(mcp, "search_exams", {"query": query})) == invalid(
            "query must be a string of 1 to 200 characters"
        )

    async def test_longest_query_is_accepted(self, mcp: FastMCP) -> None:
        ok(await call(mcp, "search_exams", {"query": "hemograma " * 20}))

    @pytest.mark.parametrize("top_k", [0, 11, -1])
    async def test_top_k_range_is_checked(self, mcp: FastMCP, top_k: int) -> None:
        assert error(await call(mcp, "search_exams", {"query": "TSH", "top_k": top_k})) == (
            invalid("top_k must be an integer from 1 to 10")
        )


class TestBatchSearch:
    async def test_results_follow_query_order(self, mcp: FastMCP, catalog: Catalog) -> None:
        result = BatchSearchResult.model_validate(
            ok(await call(mcp, "batch_search", {"queries": MENTIONS, "top_k": 3}))
        )

        assert [r.query for r in result.results] == MENTIONS
        assert result.results[0].candidates[0].code == "LAB-10001"
        assert result.results[1].candidates[0].code == "LAB-10012"
        assert all(len(r.candidates) <= 3 for r in result.results)
        assert result.catalog_sha256 == catalog.sha256

    async def test_same_answer_as_single_search(self, mcp: FastMCP) -> None:
        batch = ok(await call(mcp, "batch_search", {"queries": MENTIONS}))
        for query, item in zip(MENTIONS, batch["results"], strict=True):
            assert ok(await call(mcp, "search_exams", {"query": query})) == item

    async def test_repeated_queries_are_answered_each_time(self, mcp: FastMCP) -> None:
        result = ok(await call(mcp, "batch_search", {"queries": ["TSH", "TSH"]}))
        assert result["results"][0] == result["results"][1]

    @pytest.mark.parametrize("count", [0, 51])
    async def test_batch_size_is_checked(self, mcp: FastMCP, count: int) -> None:
        assert error(await call(mcp, "batch_search", {"queries": ["TSH"] * count})) == invalid(
            "queries must be a list of 1 to 50 strings"
        )

    async def test_largest_batch_is_accepted(self, mcp: FastMCP) -> None:
        result = ok(await call(mcp, "batch_search", {"queries": ["TSH"] * 50}))
        assert len(result["results"]) == 50

    async def test_max_batch_comes_from_settings(self, catalog: Catalog) -> None:
        config = settings(max_batch=2)
        mcp = build_server(RagService(catalog, config.max_batch), config)

        assert error(await call(mcp, "batch_search", {"queries": ["a", "b", "c"]})) == invalid(
            "queries must be a list of 1 to 2 strings"
        )

    async def test_each_query_is_checked(self, mcp: FastMCP) -> None:
        assert error(await call(mcp, "batch_search", {"queries": ["TSH", ""]})) == invalid(
            "queries[1] must be a string of 1 to 200 characters"
        )

    async def test_top_k_is_checked(self, mcp: FastMCP) -> None:
        assert error(await call(mcp, "batch_search", {"queries": ["TSH"], "top_k": 0})) == (
            invalid("top_k must be an integer from 1 to 10")
        )


class TestGetExam:
    async def test_returns_the_catalog_entry(self, mcp: FastMCP, catalog: Catalog) -> None:
        exam = Exam.model_validate(ok(await call(mcp, "get_exam", {"code": "LAB-10044"})))
        assert exam == catalog.get("LAB-10044")

    async def test_unknown_code_is_not_found(self, mcp: FastMCP) -> None:
        assert error(await call(mcp, "get_exam", {"code": "LAB-99999"})) == {
            "code": "NOT_FOUND",
            "message": "no exam with this code in the catalog",
            "retryable": False,
        }

    @pytest.mark.parametrize(
        "code", ["", "LAB-1", "lab-10001", "LAB-10001 ", "LAB-\u0661\u0660\u0660\u0660\u0661"]
    )
    async def test_malformed_code_is_invalid(self, mcp: FastMCP, code: str) -> None:
        assert error(await call(mcp, "get_exam", {"code": code})) == invalid(
            "code must look like LAB-NNNNN"
        )


class TestServiceChecks:
    """Types FastMCP would reject before the tool runs, checked again by the service."""

    @pytest.fixture
    def service(self, catalog: Catalog) -> RagService:
        return RagService(catalog, 50)

    @pytest.mark.parametrize("value", [None, 3, ["TSH"]])
    def test_query_must_be_a_string(self, service: RagService, value: object) -> None:
        with pytest.raises(Exception, match="query must be a string"):
            service.search_exams(value, 5)

    @pytest.mark.parametrize("value", [True, 2.0, "5", None])
    def test_top_k_must_be_an_integer(self, service: RagService, value: object) -> None:
        with pytest.raises(Exception, match="top_k must be an integer"):
            service.search_exams("TSH", value)

    @pytest.mark.parametrize("value", ["TSH", ("TSH",), None])
    def test_queries_must_be_a_list(self, service: RagService, value: object) -> None:
        with pytest.raises(Exception, match="queries must be a list"):
            service.batch_search(value, 5)

    def test_code_must_be_a_string(self, service: RagService) -> None:
        with pytest.raises(Exception, match="code must look like"):
            service.get_exam(10001)


class TestLogs:
    async def test_logs_counts_but_no_mentions(self, mcp: FastMCP, log_stream: io.StringIO) -> None:
        await call(mcp, "batch_search", {"queries": [*MENTIONS, ";;"]})

        logs = log_stream.getvalue()
        for word in ["Hemograma", "hemograma", "Glicemla", "crânio", "cranio"]:
            assert word not in logs
        [record] = [json.loads(line) for line in logs.splitlines()]
        assert record["event"] == "rag_done"
        assert (record["tool"], record["queries"], record["empty_results"]) == (
            "batch_search",
            5,
            1,
        )
        assert record["service"] == "rag-mcp"
        assert isinstance(record["duration_ms"], int)

    async def test_logs_under_the_callers_request_id(
        self, mcp: FastMCP, log_stream: io.StringIO
    ) -> None:
        run_id = "0b6f2c2e-4a8e-4f4c-9a51-3f0f2b8f8e11"
        await call(mcp, "batch_search", {"queries": ["TSH"], "request_id": run_id})
        await call(mcp, "search_exams", {"query": "TSH", "request_id": "Hemograma completo"})

        records = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        assert [r["request_id"] for r in records] == [run_id, None]
        assert "Hemograma" not in log_stream.getvalue()

    async def test_rejections_log_the_code_only(
        self, mcp: FastMCP, log_stream: io.StringIO
    ) -> None:
        await call(mcp, "get_exam", {"code": "LAB-99999"})

        [record] = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        assert (record["event"], record["error_code"]) == ("rag_rejected", "NOT_FOUND")
        assert "LAB-99999" not in log_stream.getvalue()

    def test_library_records_lose_their_payload(self, log_stream: io.StringIO) -> None:
        logging.getLogger("mcp.server.lowlevel.server").warning(
            "Received exception from stream: %s", "queries=['Hemograma completo']"
        )

        record = json.loads(log_stream.getvalue())
        assert record["event"] == f"Received exception from stream: {REDACTED}"
        assert "Hemograma" not in log_stream.getvalue()

    def test_own_records_are_kept(self) -> None:
        record = logging.LogRecord("rag_mcp.server", logging.INFO, __file__, 1, "a: b", None, None)
        assert LibraryLogFilter().filter(record)
        assert record.getMessage() == "a: b"


class TestHttp:
    @pytest.fixture
    def client(self, catalog: Catalog) -> Iterator[TestClient]:
        with TestClient(build_app(settings(), catalog)) as test_client:
            yield test_client

    def test_health_is_public(self, client: TestClient, catalog: Catalog) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "catalog_version": "2026.10.0",
            "catalog_sha256": catalog.sha256,
            "exam_count": 120,
        }

    def test_health_is_not_logged(self, client: TestClient, log_stream: io.StringIO) -> None:
        client.get("/health")

        # The test client's own httpx logger records the URL; only server records matter.
        lines = log_stream.getvalue().splitlines()
        assert [line for line in lines if "HTTP Request:" not in line] == []

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": f"Bearer {TOKEN}x"},
            {"Authorization": f"Basic {TOKEN}"},
            {"Authorization": TOKEN},
        ],
    )
    @pytest.mark.parametrize("path", ["/sse", "/messages/", "/unknown"])
    def test_other_paths_need_the_token(
        self, client: TestClient, headers: dict[str, str], path: str
    ) -> None:
        response = client.request("GET" if path == "/sse" else "POST", path, headers=headers)

        assert response.status_code == 401
        assert response.json() == {
            "code": "UNAUTHORIZED",
            "message": "missing or invalid bearer token",
        }
        assert response.headers["www-authenticate"] == "Bearer"

    def test_scheme_is_case_insensitive(self, client: TestClient) -> None:
        response = client.post(
            "/messages/?session_id=0" + "0" * 31, headers={"Authorization": f"bearer {TOKEN}"}
        )
        assert response.status_code != 401

    def test_duplicate_authorization_headers_are_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/messages/",
            headers=[("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {TOKEN}")],
        )
        assert response.status_code == 401

    def test_auth_failure_log_has_no_credentials(
        self, client: TestClient, log_stream: io.StringIO
    ) -> None:
        client.get("/sse", headers={"Authorization": "Bearer guessed-" + "g" * 32})

        logs = log_stream.getvalue()
        assert "guessed" not in logs
        assert any(json.loads(line)["event"] == "auth_failed" for line in logs.splitlines())


class TestMain:
    @pytest.fixture
    def env(self, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
        monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
        monkeypatch.setattr(rag_main, "configure_logging", lambda level: None)
        return monkeypatch

    def test_starts_uvicorn_with_the_loaded_catalog(
        self, env: pytest.MonkeyPatch, log_stream: io.StringIO, catalog: Catalog
    ) -> None:
        env.setenv("CATALOG_PATH", str(CATALOG_PATH))
        env.setenv("RAG_PORT", "18102")
        calls: list[dict[str, Any]] = []
        env.setattr(rag_main.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))

        services: list[str] = []
        env.setattr(rag_main.telemetry, "setup", services.append)

        rag_main.main()

        assert services == ["rag-mcp"]
        [kwargs] = calls
        assert (kwargs["host"], kwargs["port"]) == ("0.0.0.0", 18102)  # noqa: S104
        assert kwargs["log_config"] is None
        [record] = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        assert record["event"] == "catalog_loaded"
        # Logged unmasked: the hash must not read as a phone number to the redaction.
        assert record["catalog_sha256"] == catalog.sha256
        assert (record["catalog_version"], record["exam_count"]) == ("2026.10.0", 120)

    @pytest.mark.parametrize(
        "content",
        [
            None,
            "{",
            json.dumps({"catalog_version": "2026.10.0", "exams": []}),
        ],
    )
    def test_invalid_catalog_stops_before_serving(
        self,
        env: pytest.MonkeyPatch,
        tmp_path: Path,
        log_stream: io.StringIO,
        content: str | None,
    ) -> None:
        path = tmp_path / "catalog.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        env.setenv("CATALOG_PATH", str(path))
        env.setattr(rag_main.uvicorn, "run", lambda *a, **k: pytest.fail("server started"))

        with pytest.raises(SystemExit) as exit_info:
            rag_main.main()

        assert exit_info.value.code == 1
        [record] = [json.loads(line) for line in log_stream.getvalue().splitlines()]
        assert record["event"] == "catalog_invalid"
        assert record["reason"]
