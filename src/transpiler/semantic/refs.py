from collections import Counter
from collections.abc import Iterable, Iterator

from transpiler.errors import Code, Diagnostic, TranspileError, json_pointer
from transpiler.semantic.context import GraphView, SemanticContext, kind_of, node_diagnostic
from transpiler.spec.loader import read_instruction
from transpiler.spec.models import (
    AgentSpec,
    LlmExtractNode,
    McpNode,
    ScheduleNode,
    SubworkflowNode,
)

_DECLARATION_SECTIONS = ("models", "mcp_servers", "http_services", "llm_agents", "subworkflows")


def check(ctx: SemanticContext) -> list[Diagnostic]:
    return [
        *_node_refs(ctx),
        *_agent_refs(ctx),
        *_unused_declarations(ctx),
        *_unused_allowed_tools(ctx),
    ]


def _node_refs(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    for graph, node in ctx.nodes():
        if isinstance(node, McpNode):
            yield from _mcp_node_refs(spec, graph, node)
        elif isinstance(node, ScheduleNode) and node.service not in spec.http_services:
            message, hint = _unknown_reference("serviço HTTP", node.service, spec.http_services)
            yield node_diagnostic(
                graph, node, Code.E_UNKNOWN_REF, message, field="service", hint=hint
            )
        elif isinstance(node, LlmExtractNode) and node.agent not in spec.llm_agents:
            message, hint = _unknown_reference("agente LLM", node.agent, spec.llm_agents)
            yield node_diagnostic(
                graph, node, Code.E_UNKNOWN_REF, message, field="agent", hint=hint
            )
        elif isinstance(node, SubworkflowNode) and node.ref not in spec.subworkflows:
            message, hint = _unknown_reference("subworkflow", node.ref, spec.subworkflows)
            yield node_diagnostic(graph, node, Code.E_UNKNOWN_REF, message, field="ref", hint=hint)


def _mcp_node_refs(spec: AgentSpec, graph: GraphView, node: McpNode) -> Iterator[Diagnostic]:
    accepted = kind_of(node).tools
    if node.tool not in accepted:
        yield node_diagnostic(
            graph,
            node,
            Code.E_TOOL_KIND_MISMATCH,
            f'{node.kind} não sabe chamar a tool "{node.tool}"',
            field="tool",
            hint=f"tools aceitas por {node.kind}: {', '.join(sorted(accepted))}",
        )

    server = spec.mcp_servers.get(node.server)
    if server is None:
        message, hint = _unknown_reference("servidor MCP", node.server, spec.mcp_servers)
        yield node_diagnostic(graph, node, Code.E_UNKNOWN_REF, message, field="server", hint=hint)
    elif node.tool not in server.allowed_tools:
        yield node_diagnostic(
            graph,
            node,
            Code.E_TOOL_NOT_ALLOWED,
            f'A tool "{node.tool}" não está em mcp_servers.{node.server}.allowed_tools '
            f"{server.allowed_tools}.",
            field="tool",
        )


def _agent_refs(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    for name, agent in spec.llm_agents.items():
        if agent.model not in spec.models:
            message, hint = _unknown_reference("modelo", agent.model, spec.models)
            yield Diagnostic(
                code=Code.E_UNKNOWN_REF,
                message=message,
                pointer=json_pointer(("llm_agents", name, "model")),
                hint=hint,
            )

        pointer = json_pointer(("llm_agents", name, "instruction_file"))
        try:
            read_instruction(ctx.spec_dir, agent.instruction_file, pointer)
        except TranspileError as exc:
            yield from exc.diagnostics

    uses: Counter[str] = Counter()
    for graph, node in ctx.nodes():
        if not isinstance(node, LlmExtractNode):
            continue

        uses[node.agent] += 1
        if uses[node.agent] > 1:
            yield node_diagnostic(
                graph,
                node,
                Code.E_AGENT_REUSED,
                f'o agente "{node.agent}" já é usado por outro nó llm_extract',
                field="agent",
                hint="declare um agente por nó; cada um vira uma instância de Agent",
            )


def _unused_declarations(ctx: SemanticContext) -> Iterator[Diagnostic]:
    used = _used_names(ctx)
    for section in _DECLARATION_SECTIONS:
        for name in getattr(ctx.spec, section):
            if name not in used[section]:
                yield Diagnostic(
                    code=Code.W_UNUSED_DECLARATION,
                    message=f"{section}.{name} está declarado mas nenhum nó o usa",
                    pointer=json_pointer((section, name)),
                )


def _unused_allowed_tools(ctx: SemanticContext) -> Iterator[Diagnostic]:
    mcp_nodes = [node for _, node in ctx.nodes() if isinstance(node, McpNode)]
    for name, server in ctx.spec.mcp_servers.items():
        used_tools = {n.tool for n in mcp_nodes if n.server == name}
        if not used_tools:
            continue

        for index, tool in enumerate(server.allowed_tools):
            if tool not in used_tools:
                yield Diagnostic(
                    code=Code.W_UNUSED_DECLARATION,
                    message=f'a tool "{tool}" está na allowlist de {name} mas nenhum nó a usa',
                    pointer=json_pointer(("mcp_servers", name, "allowed_tools", index)),
                    hint="mantenha a allowlist mínima",
                )


def _used_names(ctx: SemanticContext) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {section: set() for section in _DECLARATION_SECTIONS}
    for _, node in ctx.nodes():
        if isinstance(node, LlmExtractNode):
            used["llm_agents"].add(node.agent)
        elif isinstance(node, McpNode):
            used["mcp_servers"].add(node.server)
        elif isinstance(node, ScheduleNode):
            used["http_services"].add(node.service)
        elif isinstance(node, SubworkflowNode):
            used["subworkflows"].add(node.ref)

    agents = ctx.spec.llm_agents
    used["models"] = {agents[name].model for name in used["llm_agents"] if name in agents}
    return used


def _unknown_reference(what: str, name: str, declared: Iterable[str]) -> tuple[str, str]:
    names = ", ".join(sorted(declared)) or "nenhum declarado"
    return f'{what} "{name}" não está declarado', f"declarados: {names}"
