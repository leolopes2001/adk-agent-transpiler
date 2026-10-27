"""References between nodes and declarations, instruction files and unused declarations."""

from collections import Counter
from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic, TranspileError, json_pointer
from transpiler.semantic.context import SemanticContext, kind_of, node_diagnostic
from transpiler.spec.loader import read_instruction
from transpiler.spec.models import (
    LexicalExtractNode,
    LlmExtractNode,
    OcrExtractNode,
    RagResolveNode,
    ScheduleNode,
    SubworkflowNode,
)

type McpNode = OcrExtractNode | LexicalExtractNode | RagResolveNode


def _unknown(what: str, name: str, known: Iterator[str] | list[str]) -> tuple[str, str]:
    names = ", ".join(sorted(known)) or "nenhum declarado"
    return f'{what} "{name}" não está declarado', f"declarados: {names}"


def _node_refs(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    for graph, node in ctx.nodes():
        if isinstance(node, OcrExtractNode | LexicalExtractNode | RagResolveNode):
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
                message, hint = _unknown("servidor MCP", node.server, list(spec.mcp_servers))
                yield node_diagnostic(
                    graph, node, Code.E_UNKNOWN_REF, message, field="server", hint=hint
                )
            elif node.tool not in server.allowed_tools:
                yield node_diagnostic(
                    graph,
                    node,
                    Code.E_TOOL_NOT_ALLOWED,
                    f'A tool "{node.tool}" não está em mcp_servers.{node.server}.allowed_tools '
                    f"{server.allowed_tools}.",
                    field="tool",
                )
        elif isinstance(node, ScheduleNode) and node.service not in spec.http_services:
            message, hint = _unknown("serviço HTTP", node.service, list(spec.http_services))
            yield node_diagnostic(
                graph, node, Code.E_UNKNOWN_REF, message, field="service", hint=hint
            )
        elif isinstance(node, LlmExtractNode) and node.agent not in spec.llm_agents:
            message, hint = _unknown("agente LLM", node.agent, list(spec.llm_agents))
            yield node_diagnostic(
                graph, node, Code.E_UNKNOWN_REF, message, field="agent", hint=hint
            )
        elif isinstance(node, SubworkflowNode) and node.ref not in spec.subworkflows:
            message, hint = _unknown("subworkflow", node.ref, list(spec.subworkflows))
            yield node_diagnostic(graph, node, Code.E_UNKNOWN_REF, message, field="ref", hint=hint)


def _agents(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    for name, agent in spec.llm_agents.items():
        if agent.model not in spec.models:
            message, hint = _unknown("modelo", agent.model, list(spec.models))
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
        if isinstance(node, LlmExtractNode):
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


def _unused(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    nodes = [node for _, node in ctx.nodes()]
    mcp_nodes: list[McpNode] = [
        n for n in nodes if isinstance(n, OcrExtractNode | LexicalExtractNode | RagResolveNode)
    ]
    agents = {n.agent for n in nodes if isinstance(n, LlmExtractNode)}
    used = {
        "models": {spec.llm_agents[a].model for a in agents if a in spec.llm_agents},
        "mcp_servers": {n.server for n in mcp_nodes},
        "http_services": {n.service for n in nodes if isinstance(n, ScheduleNode)},
        "llm_agents": agents,
        "subworkflows": {n.ref for n in nodes if isinstance(n, SubworkflowNode)},
    }
    declared: dict[str, list[str]] = {
        "models": list(spec.models),
        "mcp_servers": list(spec.mcp_servers),
        "http_services": list(spec.http_services),
        "llm_agents": list(spec.llm_agents),
        "subworkflows": list(spec.subworkflows),
    }
    for section, names in declared.items():
        for name in names:
            if name not in used[section]:
                yield Diagnostic(
                    code=Code.W_UNUSED_DECLARATION,
                    message=f"{section}.{name} está declarado mas nenhum nó o usa",
                    pointer=json_pointer((section, name)),
                )
    for name, server in spec.mcp_servers.items():
        if name not in used["mcp_servers"]:
            continue
        tools = {n.tool for n in mcp_nodes if n.server == name}
        for index, tool in enumerate(server.allowed_tools):
            if tool not in tools:
                yield Diagnostic(
                    code=Code.W_UNUSED_DECLARATION,
                    message=f'a tool "{tool}" está na allowlist de {name} mas nenhum nó a usa',
                    pointer=json_pointer(("mcp_servers", name, "allowed_tools", index)),
                    hint="mantenha a allowlist mínima",
                )


def check(ctx: SemanticContext) -> list[Diagnostic]:
    return [*_node_refs(ctx), *_agents(ctx), *_unused(ctx)]
