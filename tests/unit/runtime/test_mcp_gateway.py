import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

from agent_runtime import bootstrap
from agent_runtime.bootstrap import McpServerConfig, RuntimeDeps
from agent_runtime.errors import (
    ConfigurationError,
    PermanentError,
    PermanentToolError,
    TransientError,
    TransientToolError,
    for_http_status,
    tool_error,
)
from agent_runtime.gateways import mcp as gateway_module
from agent_runtime.gateways.mcp import (
    AdkMcpGateway,
    McpEndpoint,
    SessionMcpGateway,
    classify,
    result_body,
)

pytestmark = pytest.mark.unit

TOKEN = "unit-token-" + "x" * 32
ENDPOINT = McpEndpoint(
    url="http://rag.local:8102",
    token=TOKEN,
    allowed_tools=("batch_search",),
    connect_timeout_s=5.0,
    read_timeout_s=15.0,
)
RAG = McpServerConfig(
    url_env="RAG_MCP_URL",
    auth_token_env="MCP_AUTH_TOKEN",  # noqa: S106 - env var name
    allowed_tools=("batch_search",),
    connect_timeout_s=5.0,
    read_timeout_s=15.0,
)


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://rag.local/sse")
    return httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no connection expected")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(gateway_module, "sse_client", refuse)


class FakeSession:
    calls: list[dict[str, Any]]
    result: CallToolResult | Exception

    def __init__(self, read: Any, write: Any, read_timeout_seconds: timedelta) -> None:
        FakeSession.calls.append({"read_timeout": read_timeout_seconds})

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def initialize(self) -> None:
        FakeSession.calls.append({"initialized": True})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        FakeSession.calls.append({"tool": name, "arguments": arguments})
        if isinstance(FakeSession.result, Exception):
            raise FakeSession.result
        return FakeSession.result


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    FakeSession.calls = calls
    FakeSession.result = CallToolResult(content=[], structuredContent={"results": []})

    @asynccontextmanager
    async def fake_sse(url: str, **kwargs: Any) -> AsyncIterator[tuple[str, str]]:
        calls.append({"url": url, **kwargs})
        yield ("read", "write")

    monkeypatch.setattr(gateway_module, "sse_client", fake_sse)
    monkeypatch.setattr(gateway_module, "ClientSession", FakeSession)
    return calls


@pytest.mark.parametrize("gateway", [SessionMcpGateway, AdkMcpGateway])
async def test_tool_outside_the_allowlist_fails_before_the_network(
    gateway: type[SessionMcpGateway], no_network: None
) -> None:
    client = gateway({"rag": ENDPOINT})

    with pytest.raises(PermanentError) as unknown_tool:
        await client.call("rag", "get_exam", {"code": "LAB-10001"})
    with pytest.raises(PermanentError) as unknown_server:
        await client.call("ocr", "extract_text", {})

    assert unknown_tool.value.code == "MCP_TOOL_NOT_ALLOWED"
    assert unknown_server.value.code == "MCP_UNKNOWN_SERVER"


async def test_session_gateway_uses_the_spec_url_headers_and_timeouts(
    fake_sdk: list[dict[str, Any]],
) -> None:
    body = await SessionMcpGateway({"rag": ENDPOINT}).call(
        "rag", "batch_search", {"queries": ["TSH"], "top_k": 1}
    )

    assert body == {"results": []}
    connect, session, initialized, call = fake_sdk
    assert connect == {
        "url": "http://rag.local:8102/sse",
        "headers": {"Authorization": f"Bearer {TOKEN}"},
        "timeout": 5.0,
        "sse_read_timeout": 15.0,
    }
    assert session == {"read_timeout": timedelta(seconds=15)}
    assert initialized == {"initialized": True}
    assert call == {"tool": "batch_search", "arguments": {"queries": ["TSH"], "top_k": 1}}


async def test_session_gateway_reads_json_from_text_content(
    fake_sdk: list[dict[str, Any]],
) -> None:
    FakeSession.result = CallToolResult(content=[TextContent(type="text", text='{"a": 1}')])

    assert await SessionMcpGateway({"rag": ENDPOINT}).call("rag", "batch_search", {}) == {"a": 1}


async def test_session_gateway_classifies_failures_without_their_text(
    fake_sdk: list[dict[str, Any]],
) -> None:
    FakeSession.result = ExceptionGroup("tg", [httpx.ConnectError("secret payload 123")])

    with pytest.raises(TransientError) as caught:
        await SessionMcpGateway({"rag": ENDPOINT}).call("rag", "batch_search", {})

    assert caught.value.code == "MCP_UNAVAILABLE"
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("exc", "kind", "code"),
    [
        (_status_error(401), PermanentError, "MCP_UNAUTHORIZED"),
        (_status_error(403), PermanentError, "MCP_UNAUTHORIZED"),
        (_status_error(404), PermanentError, "MCP_HTTP_ERROR"),
        (_status_error(503), TransientError, "MCP_HTTP_ERROR"),
        (httpx.ConnectError("refused"), TransientError, "MCP_UNAVAILABLE"),
        (httpx.ReadTimeout("slow"), TransientError, "MCP_TIMEOUT"),
        (TimeoutError(), TransientError, "MCP_TIMEOUT"),
        (ConnectionRefusedError(), TransientError, "MCP_UNAVAILABLE"),
        (McpError(ErrorData(code=408, message="t")), TransientError, "MCP_TIMEOUT"),
        (McpError(ErrorData(code=-32602, message="x")), PermanentError, "MCP_PROTOCOL_ERROR"),
        (ValueError("text from the document"), PermanentError, "MCP_ERROR"),
    ],
)
def test_classify(exc: BaseException, kind: type[Exception], code: str) -> None:
    error = classify(exc, "rag")

    assert type(error) is kind
    assert error.code == code
    assert "document" not in str(error)


def test_classify_looks_inside_nested_exception_groups() -> None:
    nested = ExceptionGroup("outer", [ValueError("x"), ExceptionGroup("in", [_status_error(401)])])

    assert classify(nested, "rag").code == "MCP_UNAUTHORIZED"
    assert "ValueError" in str(classify(ExceptionGroup("g", [ValueError("x")]), "rag"))


def test_classify_keeps_runtime_errors() -> None:
    error = TransientError("X")

    assert classify(ExceptionGroup("g", [error]), "rag") is error


@pytest.mark.parametrize(
    ("body", "kind", "tool_code"),
    [
        ({"code": "BUSY", "message": "retry later", "retryable": True}, TransientToolError, "BUSY"),
        (
            {"code": "INVALID_ARGUMENT", "message": "m", "retryable": False},
            PermanentToolError,
            "INVALID_ARGUMENT",
        ),
        ({"code": "BUSY", "retryable": "yes"}, PermanentToolError, "BUSY"),
        ({"code": "not an id!", "retryable": True}, TransientToolError, "UNKNOWN"),
        ({}, PermanentToolError, "UNKNOWN"),
    ],
)
def test_tool_errors(body: dict[str, Any], kind: type[Exception], tool_code: str) -> None:
    with pytest.raises(kind) as caught:
        result_body("ocr", "extract_text", {"structuredContent": body, "isError": True})

    assert caught.value.tool_code == tool_code  # type: ignore[attr-defined]
    assert "retry later" not in str(caught.value)


def test_tool_error_body_may_come_as_text() -> None:
    result = {"content": [{"type": "text", "text": '{"code": "BUSY", "retryable": true}'}]}

    with pytest.raises(TransientToolError):
        result_body("ocr", "extract_text", {**result, "isError": True})


@pytest.mark.parametrize(
    "result",
    [
        {"content": []},
        {"content": [{"type": "text", "text": "not json"}]},
        {"content": [{"type": "text", "text": "[1, 2]"}]},
        {"content": [{"type": "image"}]},
        {"content": "x"},
    ],
)
def test_result_without_a_json_object_is_permanent(result: dict[str, Any]) -> None:
    with pytest.raises(PermanentError, match="MCP_INVALID_RESULT"):
        result_body("rag", "batch_search", result)


def test_error_helpers() -> None:
    assert isinstance(for_http_status(502, "X"), TransientError)
    assert isinstance(for_http_status(429, "X"), TransientError)
    assert type(for_http_status(422, "X")) is PermanentError
    assert isinstance(tool_error("s", "t", {"code": "BUSY", "retryable": True}), TransientError)


def test_endpoint_repr_hides_the_token() -> None:
    assert TOKEN not in repr(ENDPOINT)
    assert ENDPOINT.sse_url == "http://rag.local:8102/sse"
    assert McpEndpoint("http://h/sse", "t", (), 1.0, 1.0).sse_url == "http://h/sse"


def test_runtime_deps_reads_env_only_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_MCP_URL", raising=False)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    deps = RuntimeDeps.lazy(mcp_servers={"rag": RAG})

    assert deps.required_env() == ["MCP_AUTH_TOKEN", "RAG_MCP_URL"]
    assert deps.missing_env() == ["MCP_AUTH_TOKEN", "RAG_MCP_URL"]
    with pytest.raises(ConfigurationError, match="RAG_MCP_URL, MCP_AUTH_TOKEN"):
        _ = deps.mcp

    monkeypatch.setenv("RAG_MCP_URL", "http://rag.local:8102")
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    gateway = deps.mcp

    assert isinstance(gateway, SessionMcpGateway)
    assert deps.mcp is gateway
    assert deps.missing_env() == []


@pytest.mark.parametrize(("choice", "kind"), [("adk", AdkMcpGateway), ("", SessionMcpGateway)])
def test_runtime_deps_gateway_choice(
    monkeypatch: pytest.MonkeyPatch, choice: str, kind: type[object]
) -> None:
    monkeypatch.setenv("RAG_MCP_URL", "http://rag.local:8102")
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv(bootstrap.MCP_GATEWAY_ENV, choice)

    assert isinstance(RuntimeDeps.lazy(mcp_servers={"rag": RAG}).mcp, kind)


def test_runtime_deps_rejects_an_unknown_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bootstrap.MCP_GATEWAY_ENV, "grpc")

    with pytest.raises(ConfigurationError, match="INVALID_ENV"):
        _ = RuntimeDeps.lazy(mcp_servers={"rag": RAG}).mcp


async def test_runtime_deps_aclose_rebuilds_but_keeps_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MCP_URL", "http://rag.local:8102")
    monkeypatch.setenv("MCP_AUTH_TOKEN", TOKEN)
    deps = RuntimeDeps.lazy(mcp_servers={"rag": RAG})
    first = deps.mcp

    await deps.aclose()
    assert deps.mcp is not first

    injected = SessionMcpGateway({})
    deps.use(mcp=injected)
    await deps.aclose()
    assert deps.mcp is injected


async def test_adk_gateway_aclose_closes_its_toolsets() -> None:
    gateway = AdkMcpGateway({"rag": ENDPOINT})
    toolset = gateway._toolset("rag", ENDPOINT)

    assert gateway._toolset("rag", ENDPOINT) is toolset
    await gateway.aclose()
    assert gateway._toolset("rag", ENDPOINT) is not toolset
