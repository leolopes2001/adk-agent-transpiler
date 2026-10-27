"""Runtime dependencies declared by generated code.

Generated modules are imported by the transpiler's validation probe and by
tests, so building the graph must never open a connection. `RuntimeDeps.lazy`
only records configuration; gateways are created on first use (phase 06).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    url_env: str
    auth_token_env: str
    allowed_tools: tuple[str, ...]
    connect_timeout_s: float
    read_timeout_s: float


@dataclass(frozen=True, slots=True)
class HttpServiceConfig:
    base_url_env: str
    api_key_env: str
    timeout_s: float


class RuntimeDeps:
    """Configuration of the MCP servers and HTTP services a workflow uses."""

    def __init__(
        self,
        mcp_servers: Mapping[str, McpServerConfig],
        http_services: Mapping[str, HttpServiceConfig],
    ) -> None:
        self._mcp_servers = MappingProxyType(dict(mcp_servers))
        self._http_services = MappingProxyType(dict(http_services))

    @classmethod
    def lazy(
        cls,
        *,
        mcp_servers: Mapping[str, McpServerConfig] | None = None,
        http_services: Mapping[str, HttpServiceConfig] | None = None,
    ) -> RuntimeDeps:
        return cls(mcp_servers or {}, http_services or {})

    @property
    def mcp_servers(self) -> Mapping[str, McpServerConfig]:
        return self._mcp_servers

    @property
    def http_services(self) -> Mapping[str, HttpServiceConfig]:
        return self._http_services
