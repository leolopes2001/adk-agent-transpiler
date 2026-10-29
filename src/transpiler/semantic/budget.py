"""A node timeout shorter than one transport attempt retries over a request still in flight."""

from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic
from transpiler.semantic.context import SemanticContext, node_diagnostic
from transpiler.spec.models import AgentSpec, McpNode, ScheduleNode


def check(ctx: SemanticContext) -> list[Diagnostic]:
    return list(_timeout_budgets(ctx))


def _timeout_budgets(ctx: SemanticContext) -> Iterator[Diagnostic]:
    default = ctx.spec.policies.defaults.timeout_s
    for graph, node in ctx.nodes():
        if not isinstance(node, McpNode | ScheduleNode):
            continue

        transport = _transport_timeout(ctx.spec, node)
        if transport is None:
            continue

        needed, setting = transport
        timeout = default if node.timeout_s is None else node.timeout_s
        if timeout < needed:
            source = "timeout_s" if node.timeout_s is not None else "o timeout padrão"
            yield node_diagnostic(
                graph,
                node,
                Code.E_TIMEOUT_BUDGET,
                f"{source} do nó ({timeout:g}s) é menor que {setting} ({needed:g}s)",
                field="timeout_s" if node.timeout_s is not None else None,
                hint=f"use timeout_s >= {needed:g}",
            )


def _transport_timeout(spec: AgentSpec, node: McpNode | ScheduleNode) -> tuple[float, str] | None:
    if isinstance(node, ScheduleNode):
        service = spec.http_services.get(node.service)
        if service is None:
            return None

        return service.timeout_s, f"http_services.{node.service}.timeout_s"

    server = spec.mcp_servers.get(node.server)
    if server is None:
        return None

    return server.read_timeout_s, f"mcp_servers.{node.server}.read_timeout_s"
