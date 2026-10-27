"""Runtime dependencies declared by generated code.

Generated modules are imported by the transpiler's validation probe and by
tests, so building the graph must never open a connection or read a secret.
`RuntimeDeps.lazy` only records configuration; the environment is read and
the gateways are created the first time a node uses them, and `aclose()`
releases them at the end of a run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent_runtime.errors import ConfigurationError
from agent_runtime.gateways.mcp import AdkMcpGateway, McpEndpoint, McpGateway, SessionMcpGateway

MCP_GATEWAY_ENV = "AGENT_MCP_GATEWAY"
"""`session` (default) or `adk`; see ADR-0009."""

_MCP_GATEWAYS: Mapping[str, type[SessionMcpGateway | AdkMcpGateway]] = MappingProxyType(
    {"session": SessionMcpGateway, "adk": AdkMcpGateway}
)


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
        self._mcp: McpGateway | None = None
        self._injected_mcp: McpGateway | None = None

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

    def required_env(self) -> list[str]:
        """Environment variables the declared servers and services need."""
        names = {n for c in self._mcp_servers.values() for n in (c.url_env, c.auth_token_env)}
        names |= {n for c in self._http_services.values() for n in (c.base_url_env, c.api_key_env)}
        return sorted(names)

    def missing_env(self) -> list[str]:
        return [name for name in self.required_env() if not os.environ.get(name)]

    def _env(self, *names: str) -> list[str]:
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise ConfigurationError("MISSING_ENV", ", ".join(missing))
        return [os.environ[name] for name in names]

    @property
    def mcp(self) -> McpGateway:
        """Gateway to the declared MCP servers; built from the environment on first use."""
        if self._injected_mcp is not None:
            return self._injected_mcp
        if self._mcp is None:
            self._mcp = self._build_mcp()
        return self._mcp

    def _build_mcp(self) -> McpGateway:
        choice = os.environ.get(MCP_GATEWAY_ENV) or "session"
        gateway = _MCP_GATEWAYS.get(choice)
        if gateway is None:
            raise ConfigurationError(
                "INVALID_ENV", f"{MCP_GATEWAY_ENV} must be one of {sorted(_MCP_GATEWAYS)}"
            )
        self._env(*(n for c in self._mcp_servers.values() for n in (c.url_env, c.auth_token_env)))
        endpoints = {}
        for name, config in self._mcp_servers.items():
            url, token = self._env(config.url_env, config.auth_token_env)
            endpoints[name] = McpEndpoint(
                url=url,
                token=token,
                allowed_tools=config.allowed_tools,
                connect_timeout_s=config.connect_timeout_s,
                read_timeout_s=config.read_timeout_s,
            )
        return gateway(endpoints)

    def use(self, *, mcp: McpGateway | None = None) -> None:
        """Replace gateways with given ones (tests, fakes); `aclose` keeps them."""
        self._injected_mcp = mcp

    async def aclose(self) -> None:
        """Release the gateways built from the environment; the next use rebuilds them."""
        mcp, self._mcp = self._mcp, None
        if mcp is not None:
            await mcp.aclose()
