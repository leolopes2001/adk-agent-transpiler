from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from agent_runtime.errors import ConfigurationError
from agent_runtime.gateways.mcp import AdkMcpGateway, McpEndpoint, McpGateway, SessionMcpGateway
from agent_runtime.gateways.scheduler import SchedulerClient, SchedulerGateway

MCP_GATEWAY_ENV = "AGENT_MCP_GATEWAY"
INPUT_ROOT_ENV = "INPUT_ROOT"

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


# The transpiler probe and the tests import generated modules: no env read before first use.
class RuntimeDeps:
    def __init__(
        self,
        mcp_servers: Mapping[str, McpServerConfig],
        http_services: Mapping[str, HttpServiceConfig],
    ) -> None:
        self._mcp_servers = MappingProxyType(dict(mcp_servers))
        self._http_services = MappingProxyType(dict(http_services))
        self._mcp: McpGateway | None = None
        self._injected_mcp: McpGateway | None = None
        self._schedulers: dict[str, SchedulerGateway] = {}
        self._injected_schedulers: dict[str, SchedulerGateway] = {}

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

    @property
    def mcp(self) -> McpGateway:
        if self._injected_mcp is not None:
            return self._injected_mcp

        if self._mcp is None:
            self._mcp = self._build_mcp()
        return self._mcp

    def required_env(self) -> list[str]:
        names = set(self._mcp_env_names())
        names |= {n for c in self._http_services.values() for n in (c.base_url_env, c.api_key_env)}
        return sorted(names)

    def missing_env(self) -> list[str]:
        return [name for name in self.required_env() if not os.environ.get(name)]

    def scheduler(self, service: str) -> SchedulerGateway:
        if service in self._injected_schedulers:
            return self._injected_schedulers[service]

        client = self._schedulers.get(service)
        if client is None:
            config = self._http_services.get(service)
            if config is None:
                raise ConfigurationError("UNKNOWN_SERVICE", f"service {service!r} is not declared")

            base_url, api_key = self._require_env(config.base_url_env, config.api_key_env)
            client = SchedulerClient(base_url, api_key, config.timeout_s)
            self._schedulers[service] = client

        return client

    def use(
        self,
        *,
        mcp: McpGateway | None = None,
        schedulers: Mapping[str, SchedulerGateway] | None = None,
    ) -> None:
        """Replace the gateways (tests, fakes); `aclose` leaves these open."""
        self._injected_mcp = mcp
        self._injected_schedulers = dict(schedulers or {})

    async def aclose(self) -> None:
        mcp, self._mcp = self._mcp, None
        schedulers, self._schedulers = self._schedulers, {}
        if mcp is not None:
            await mcp.aclose()
        for client in schedulers.values():
            await client.aclose()

    def _build_mcp(self) -> McpGateway:
        choice = os.environ.get(MCP_GATEWAY_ENV) or "session"
        gateway = _MCP_GATEWAYS.get(choice)
        if gateway is None:
            raise ConfigurationError(
                "INVALID_ENV", f"{MCP_GATEWAY_ENV} must be one of {sorted(_MCP_GATEWAYS)}"
            )

        # Checked all at once first, so the error lists every missing name.
        self._require_env(*self._mcp_env_names())
        endpoints = {}
        for name, config in self._mcp_servers.items():
            url, token = self._require_env(config.url_env, config.auth_token_env)
            endpoints[name] = McpEndpoint(
                url=url,
                token=token,
                allowed_tools=config.allowed_tools,
                connect_timeout_s=config.connect_timeout_s,
                read_timeout_s=config.read_timeout_s,
            )

        return gateway(endpoints)

    def _mcp_env_names(self) -> list[str]:
        return [n for c in self._mcp_servers.values() for n in (c.url_env, c.auth_token_env)]

    def _require_env(self, *names: str) -> list[str]:
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise ConfigurationError("MISSING_ENV", ", ".join(missing))

        return [os.environ[name] for name in names]


def input_root() -> Path:
    return Path(os.environ.get(INPUT_ROOT_ENV) or Path.cwd()).resolve()
