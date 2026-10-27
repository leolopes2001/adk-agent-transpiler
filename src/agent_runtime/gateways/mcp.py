"""Gateway the deterministic nodes use to call MCP tools (ADR-0004, ADR-0007).

The model never sees these tools: `ocr`, `lexical_extract` and `resolve` call
them with arguments built in code. Both implementations enforce the spec's
allowlist before touching the network, apply the spec's timeouts, never retry
(the node's RetryConfig is the only retry level) and turn every failure into
a `TransientError` or `PermanentError` whose message carries no payload.

`SessionMcpGateway` is the default (ADR-0009): one SSE connection per call
through the `mcp` SDK. `AdkMcpGateway` goes through ADK's `McpToolset`; it is
kept as an alternative because ADK retries session setup once internally.
"""

import json
from collections.abc import Iterator, Mapping
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

from agent_runtime.errors import (
    AgentRuntimeError,
    PermanentError,
    TransientError,
    for_http_status,
    tool_error,
)


class McpGateway(Protocol):
    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        """Result body of the tool; raises TransientError or PermanentError."""
        ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class McpEndpoint:
    url: str
    """Base URL of the server; the SSE stream is at `<url>/sse`."""
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


def _leaves(exc: BaseException) -> Iterator[BaseException]:
    """The exceptions inside (possibly nested) exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def classify(exc: BaseException, server: str) -> AgentRuntimeError:  # noqa: PLR0911 - one per failure class
    """Runtime error for a failure raised while talking to `server`.

    Task groups in the SSE client wrap the real cause in exception groups; the
    first leaf that says something about the transport decides.
    """
    leaves = list(_leaves(exc))
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
    """JSON body of a dumped `CallToolResult`; tool errors are raised."""
    body = result.get("structuredContent")
    if not isinstance(body, Mapping):
        body = _text_json(result.get("content"))
    if result.get("isError") is True:
        raise tool_error(server, tool, body if isinstance(body, Mapping) else {})
    if not isinstance(body, Mapping):
        raise PermanentError("MCP_INVALID_RESULT", f"{server}.{tool} returned no JSON object")
    return body


def _text_json(content: object) -> object:
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


class _AllowlistGateway:
    def __init__(self, endpoints: Mapping[str, McpEndpoint]) -> None:
        self._endpoints = dict(endpoints)

    def _endpoint(self, server: str, tool: str) -> McpEndpoint:
        endpoint = self._endpoints.get(server)
        if endpoint is None:
            raise PermanentError("MCP_UNKNOWN_SERVER", f"server {server!r} is not declared")
        if tool not in endpoint.allowed_tools:
            raise PermanentError("MCP_TOOL_NOT_ALLOWED", f"{server}.{tool} is not in the allowlist")
        return endpoint


class SessionMcpGateway(_AllowlistGateway):
    """`mcp.ClientSession` over SSE, one connection per call."""

    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
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
                result: CallToolResult = await session.call_tool(tool, dict(args))
        except Exception as exc:  # noqa: BLE001 - every failure is classified, none leaks
            raise classify(exc, server) from None
        return result_body(server, tool, _dump(result))

    async def aclose(self) -> None:
        """Nothing stays open between calls."""


def _dump(result: CallToolResult) -> dict[str, Any]:
    content = [
        {"type": "text", "text": item.text} if isinstance(item, TextContent) else {}
        for item in result.content
    ]
    return {
        "content": content,
        "structuredContent": result.structuredContent,
        "isError": result.isError,
    }


class AdkMcpGateway(_AllowlistGateway):
    """ADK `McpToolset` per server, created on first use, with `tool_filter` = allowlist.

    Tools are invoked directly, without an agent. `McpTool.run_async` declares a
    `tool_context`; with no auth scheme, header provider or confirmation it is
    not read, so None is passed (pinned by the ADK contract test).
    """

    def __init__(self, endpoints: Mapping[str, McpEndpoint]) -> None:
        super().__init__(endpoints)
        self._toolsets: dict[str, McpToolset] = {}

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

    async def call(self, server: str, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        endpoint = self._endpoint(server, tool)
        toolset = self._toolset(server, endpoint)
        try:
            tools = {t.name: t for t in await toolset.get_tools()}
            if tool not in tools:
                raise PermanentError("MCP_TOOL_NOT_FOUND", f"{server} does not offer {tool}")
            result = await tools[tool].run_async(args=dict(args), tool_context=cast(Any, None))
        except Exception as exc:  # noqa: BLE001 - every failure is classified, none leaks
            raise classify(exc, server) from None
        if not isinstance(result, Mapping):
            raise PermanentError("MCP_INVALID_RESULT", f"{server}.{tool} returned no JSON object")
        if "error" in result and "content" not in result:
            # ADK's graceful error mode turns transport failures into {"error": text}.
            raise TransientError("MCP_UNAVAILABLE", f"{server} call failed")
        return result_body(server, tool, result)

    async def aclose(self) -> None:
        toolsets, self._toolsets = self._toolsets, {}
        for toolset in toolsets.values():
            await toolset.close()
