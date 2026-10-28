import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol, cast

import httpx
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from opentelemetry.trace import SpanKind

from agent_runtime.errors import (
    AgentRuntimeError,
    PermanentError,
    TransientError,
    for_http_status,
    tool_error,
)
from lab_common import telemetry as lab_telemetry

tracer = lab_telemetry.tracer(__name__)


class McpGateway(Protocol):
    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class McpEndpoint:
    url: str
    token: str
    allowed_tools: tuple[str, ...]
    connect_timeout_s: float
    read_timeout_s: float

    def __repr__(self) -> str:
        return f"McpEndpoint(url={self.url!r}, allowed_tools={self.allowed_tools!r})"

    @property
    def sse_url(self) -> str:
        return self.url if self.url.rstrip("/").endswith("/sse") else self.url.rstrip("/") + "/sse"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


# Never retries (the node's RetryConfig is the only retry level); errors never carry a payload.
class _AllowlistGateway:
    def __init__(self, endpoints: Mapping[str, McpEndpoint]) -> None:
        self._endpoints = dict(endpoints)

    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        with _client_span(server, tool):
            return await self._call(server, tool, args)

    async def _call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _endpoint(self, server: str, tool: str) -> McpEndpoint:
        endpoint = self._endpoints.get(server)
        if endpoint is None:
            raise PermanentError("MCP_UNKNOWN_SERVER", f"server {server!r} is not declared")

        if tool not in endpoint.allowed_tools:
            raise PermanentError("MCP_TOOL_NOT_ALLOWED", f"{server}.{tool} is not in the allowlist")

        return endpoint


# One SSE connection per call: nothing stays open between calls.
class SessionMcpGateway(_AllowlistGateway):
    async def aclose(self) -> None:
        pass

    async def _call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        endpoint = self._endpoint(server, tool)
        read_timeout = timedelta(seconds=endpoint.read_timeout_s)
        try:
            async with (
                sse_client(
                    endpoint.sse_url,
                    headers=endpoint.headers,
                    timeout=endpoint.connect_timeout_s,
                    sse_read_timeout=endpoint.read_timeout_s,
                ) as (read, write),
                ClientSession(read, write, read_timeout_seconds=read_timeout) as session,
            ):
                await session.initialize()
                # The traceparent in _meta joins the server's span to the run (ADK's toolset can't).
                result: CallToolResult = await session.call_tool(
                    tool, dict(args), meta=lab_telemetry.trace_meta() or None
                )
        except Exception as exc:  # noqa: BLE001 - every failure is classified, none leaks
            raise classify(exc, server) from None

        return result_body(server, tool, _result_dict(result))


# Kept as an alternative to SessionMcpGateway: ADK retries session setup once internally.
class AdkMcpGateway(_AllowlistGateway):
    def __init__(self, endpoints: Mapping[str, McpEndpoint]) -> None:
        super().__init__(endpoints)
        self._toolsets: dict[str, McpToolset] = {}

    async def aclose(self) -> None:
        toolsets, self._toolsets = self._toolsets, {}
        for toolset in toolsets.values():
            await toolset.close()

    async def _call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        endpoint = self._endpoint(server, tool)
        toolset = self._toolset(server, endpoint)
        try:
            tools = {t.name: t for t in await toolset.get_tools()}
            if tool not in tools:
                raise PermanentError("MCP_TOOL_NOT_FOUND", f"{server} does not offer {tool}")

            # run_async reads tool_context only for auth, header providers or confirmation.
            result = await tools[tool].run_async(args=dict(args), tool_context=cast(Any, None))
        except Exception as exc:  # noqa: BLE001 - every failure is classified, none leaks
            raise classify(exc, server) from None

        if not isinstance(result, Mapping):
            raise PermanentError("MCP_INVALID_RESULT", f"{server}.{tool} returned no JSON object")

        if "error" in result and "content" not in result:
            # ADK's graceful error mode turns transport failures into {"error": text}.
            raise TransientError("MCP_UNAVAILABLE", f"{server} call failed")

        return result_body(server, tool, result)

    def _toolset(self, server: str, endpoint: McpEndpoint) -> McpToolset:
        toolset = self._toolsets.get(server)
        if toolset is None:
            toolset = McpToolset(
                connection_params=SseConnectionParams(
                    url=endpoint.sse_url,
                    headers=endpoint.headers,
                    timeout=endpoint.connect_timeout_s,
                    sse_read_timeout=endpoint.read_timeout_s,
                ),
                tool_filter=list(endpoint.allowed_tools),
            )
            self._toolsets[server] = toolset

        return toolset


# The SSE client's task groups wrap the real cause; the first leaf about the transport decides.
def classify(exc: BaseException, server: str) -> AgentRuntimeError:  # noqa: PLR0911 - one per failure class
    leaves = list(_leaf_exceptions(exc))
    for leaf in leaves:
        if isinstance(leaf, AgentRuntimeError):
            return leaf

        if isinstance(leaf, httpx.HTTPStatusError):
            status = leaf.response.status_code
            if status in (401, 403):
                return PermanentError("MCP_UNAUTHORIZED", f"{server} rejected the credentials")

            return for_http_status(status, "MCP_HTTP_ERROR")

        if isinstance(leaf, httpx.TimeoutException | TimeoutError):
            return TransientError("MCP_TIMEOUT", f"{server} did not answer in time")

        if isinstance(leaf, McpError):
            if leaf.error.code == httpx.codes.REQUEST_TIMEOUT:
                return TransientError("MCP_TIMEOUT", f"{server} did not answer in time")

            return PermanentError("MCP_PROTOCOL_ERROR", f"{server} returned a protocol error")

        if isinstance(leaf, httpx.TransportError | ConnectionError | OSError):
            return TransientError("MCP_UNAVAILABLE", f"{server} is unreachable")

    names = ",".join(sorted({type(leaf).__name__ for leaf in leaves}))
    return PermanentError("MCP_ERROR", f"{server} call failed ({names})")


def result_body(server: str, tool: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    body = result.get("structuredContent")
    if not isinstance(body, Mapping):
        body = _single_text_json(result.get("content"))
    if result.get("isError") is True:
        raise tool_error(server, tool, body if isinstance(body, Mapping) else {})

    if not isinstance(body, Mapping):
        raise PermanentError("MCP_INVALID_RESULT", f"{server}.{tool} returned no JSON object")

    return body


def _leaf_exceptions(exc: BaseException) -> Iterator[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaf_exceptions(inner)
    else:
        yield exc


def _single_text_json(content: object) -> object:
    if not isinstance(content, list) or len(content) != 1:
        return None

    [item] = content
    text = item.get("text") if isinstance(item, Mapping) else None
    if not isinstance(text, str):
        return None

    try:
        return json.loads(text)
    except ValueError:
        return None


def _result_dict(result: CallToolResult) -> dict[str, Any]:
    content = [
        {"type": "text", "text": item.text} if isinstance(item, TextContent) else {}
        for item in result.content
    ]
    return {
        "content": content,
        "structuredContent": result.structuredContent,
        "isError": result.isError,
    }


@contextmanager
def _client_span(server: str, tool: str) -> Iterator[None]:
    with tracer.start_as_current_span(
        f"mcp {server}.{tool}",
        kind=SpanKind.CLIENT,
        attributes={
            "mcp.method.name": "tools/call",
            "gen_ai.tool.name": tool,
            "lab.mcp_server": server,
        },
        record_exception=False,
    ) as span:
        try:
            yield
        except AgentRuntimeError as error:
            lab_telemetry.annotate(tool_error_code=getattr(error, "tool_code", None))
            lab_telemetry.fail(error.code, span)
            raise
