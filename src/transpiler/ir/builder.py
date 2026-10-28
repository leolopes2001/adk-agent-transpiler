"""Trusts the semantic pass: a bad reference here is a transpiler bug, so it raises KeyError."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from pydantic import BaseModel

from transpiler.ir.model import (
    EdgeIR,
    GraphIR,
    HttpServiceIR,
    LlmAgentIR,
    McpServerIR,
    ModelIR,
    NodeIR,
    ParamValue,
    ProgramIR,
    RetryIR,
)
from transpiler.registry import kind_spec
from transpiler.spec.loader import InstructionFile
from transpiler.spec.models import (
    AgentSpec,
    GraphSpec,
    LlmExtractNode,
    McpNode,
    NodeSpec,
    RetrySpec,
    ScheduleNode,
    SubworkflowNode,
)


class _Builder:
    def __init__(self, spec: AgentSpec, instructions: Mapping[str, InstructionFile]) -> None:
        self.spec = spec
        self.instructions = instructions
        self.used_servers: set[str] = set()
        self.used_services: set[str] = set()
        self.needs_resumability = False

    def server(self, key: str) -> McpServerIR:
        server = self.spec.mcp_servers[key]
        self.used_servers.add(key)
        return McpServerIR(
            key=key,
            url_env=server.url_env,
            auth_token_env=server.auth_token_env,
            allowed_tools=tuple(server.allowed_tools),
            connect_timeout_s=float(server.connect_timeout_s),
            read_timeout_s=float(server.read_timeout_s),
        )

    def service(self, key: str) -> HttpServiceIR:
        service = self.spec.http_services[key]
        self.used_services.add(key)
        return HttpServiceIR(
            key=key,
            base_url_env=service.base_url_env,
            api_key_env=service.api_key_env,
            timeout_s=float(service.timeout_s),
        )

    def agent(self, key: str) -> LlmAgentIR:
        agent = self.spec.llm_agents[key]
        model = self.spec.models[agent.model]
        instruction = self.instructions[key]
        return LlmAgentIR(
            key=key,
            model=ModelIR(
                key=agent.model,
                provider=model.provider,
                name=model.name,
                temperature=float(model.temperature),
                max_output_tokens=model.max_output_tokens,
            ),
            instruction=instruction.text,
            instruction_sha256=instruction.sha256,
            output_schema=agent.output_schema,
        )

    def build_node(self, node: NodeSpec) -> NodeIR:
        spec = kind_spec(node.kind)
        retry, timeout = self._retry_and_timeout(node) if spec.accepts_retry else (None, None)
        if spec.rerun_on_resume:
            self.needs_resumability = True

        result = NodeIR(
            id=node.id,
            kind=spec.kind,
            params=MappingProxyType(_params(node)),
            retry=retry,
            timeout_s=timeout,
        )
        return self._with_references(result, node)

    def build_graph(self, name: str, graph: GraphSpec) -> GraphIR:
        return GraphIR(
            name=name,
            nodes=tuple(self.build_node(node) for node in graph.nodes),
            edges=tuple(EdgeIR(e.source, e.target, e.route) for e in graph.edges),
        )

    def _retry_and_timeout(self, node: NodeSpec) -> tuple[RetryIR, float]:
        defaults = self.spec.policies.defaults
        chosen: RetrySpec = getattr(node, "retry", None) or defaults.retry
        node_timeout = getattr(node, "timeout_s", None)
        retry = RetryIR(
            max_attempts=chosen.max_attempts,
            initial_delay_s=float(chosen.initial_delay_s),
            max_delay_s=float(chosen.max_delay_s),
        )
        return retry, float(defaults.timeout_s if node_timeout is None else node_timeout)

    def _with_references(self, result: NodeIR, node: NodeSpec) -> NodeIR:
        if isinstance(node, McpNode):
            return replace(result, server=self.server(node.server), tool=node.tool)

        if isinstance(node, ScheduleNode):
            return replace(result, service=self.service(node.service))

        if isinstance(node, LlmExtractNode):
            return replace(result, agent=self.agent(node.agent))

        if isinstance(node, SubworkflowNode):
            subgraph = self.build_graph(node.ref, self.spec.subworkflows[node.ref])
            return replace(result, subgraph=subgraph)

        return result


def build_program(
    spec: AgentSpec, *, spec_sha256: str, instructions: Mapping[str, InstructionFile]
) -> ProgramIR:
    builder = _Builder(spec, instructions)
    root = builder.build_graph(spec.agent.name, spec.workflow)
    return ProgramIR(
        agent_name=spec.agent.name,
        description=spec.agent.description,
        version=spec.agent.version,
        root=root,
        servers=tuple(builder.server(k) for k in spec.mcp_servers if k in builder.used_servers),
        services=tuple(
            builder.service(k) for k in spec.http_services if k in builder.used_services
        ),
        needs_resumability=builder.needs_resumability,
        spec_sha256=spec_sha256,
    )


def _params(node: NodeSpec) -> dict[str, ParamValue]:
    node_params: BaseModel | None = getattr(node, "params", None)
    return {} if node_params is None else node_params.model_dump()
