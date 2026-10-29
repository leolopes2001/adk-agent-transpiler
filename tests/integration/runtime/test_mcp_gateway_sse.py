"""Both MCP gateways against the real RAG server over SSE."""

import socket

import pytest

from agent_runtime.errors import PermanentError, PermanentToolError, TransientError
from agent_runtime.gateways.mcp import AdkMcpGateway, McpEndpoint, SessionMcpGateway
from lab_common.contracts.rag import BatchSearchResult

pytestmark = pytest.mark.integration

GATEWAYS = [SessionMcpGateway, AdkMcpGateway]


@pytest.fixture
def endpoint(rag_url: str, mcp_token: str) -> McpEndpoint:
    return _endpoint(rag_url, mcp_token)


def _endpoint(url: str, token: str) -> McpEndpoint:
    return McpEndpoint(
        url=url,
        token=token,
        allowed_tools=("batch_search",),
        connect_timeout_s=5.0,
        read_timeout_s=15.0,
    )


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@pytest.mark.parametrize("gateway", GATEWAYS)
async def test_batch_search_returns_the_contract(
    endpoint: McpEndpoint, gateway: type[SessionMcpGateway]
) -> None:
    client = gateway({"rag": endpoint})
    try:
        body = await client.call(
            "rag", "batch_search", {"queries": ["1. Hemograma completo", "TSH"], "top_k": 2}
        )
        again = await client.call("rag", "batch_search", {"queries": ["PCR"], "top_k": 1})
    finally:
        await client.aclose()

    result = BatchSearchResult.model_validate(body)
    assert [r.candidates[0].code for r in result.results] == ["LAB-10001", "LAB-10043"]
    assert BatchSearchResult.model_validate(again).results[0].candidates[0].code == "LAB-10037"


@pytest.mark.parametrize("gateway", GATEWAYS)
async def test_tool_error_is_permanent_with_its_code(
    endpoint: McpEndpoint, gateway: type[SessionMcpGateway]
) -> None:
    client = gateway({"rag": endpoint})
    try:
        with pytest.raises(PermanentToolError) as caught:
            await client.call("rag", "batch_search", {"queries": [], "top_k": 1})
    finally:
        await client.aclose()

    assert caught.value.tool_code == "INVALID_ARGUMENT"


async def test_wrong_token_is_permanent(rag_url: str) -> None:
    client = SessionMcpGateway({"rag": _endpoint(rag_url, token="wrong-" + "x" * 32)})

    with pytest.raises(PermanentError) as caught:
        await client.call("rag", "batch_search", {"queries": ["TSH"], "top_k": 1})

    assert caught.value.code == "MCP_UNAUTHORIZED"


async def test_adk_gateway_cannot_tell_a_wrong_token_from_an_outage(rag_url: str) -> None:
    """ADK wraps session failures in ConnectionError (ADR-0009)."""
    client = AdkMcpGateway({"rag": _endpoint(rag_url, token="wrong-" + "x" * 32)})
    try:
        with pytest.raises(TransientError):
            await client.call("rag", "batch_search", {"queries": ["TSH"], "top_k": 1})
    finally:
        await client.aclose()


@pytest.mark.parametrize("gateway", GATEWAYS)
async def test_unreachable_server_is_transient(gateway: type[SessionMcpGateway]) -> None:
    client = gateway({"rag": _endpoint(f"http://127.0.0.1:{_closed_port()}", "t" * 32)})
    try:
        with pytest.raises(TransientError) as caught:
            await client.call("rag", "batch_search", {"queries": ["TSH"], "top_k": 1})
    finally:
        await client.aclose()

    assert caught.value.code == "MCP_UNAVAILABLE"
