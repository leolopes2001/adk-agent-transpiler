"""Timeout budget: a node must wait at least as long as one attempt of the call it makes.

Otherwise the node times out while the transport is still within its own
limit, and the retry fires on top of a request that may still succeed.
"""

from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic
from transpiler.semantic.context import SemanticContext, node_diagnostic
from transpiler.spec.models import LexicalExtractNode, OcrExtractNode, RagResolveNode, ScheduleNode


def _budgets(ctx: SemanticContext) -> Iterator[Diagnostic]:
    spec = ctx.spec
    default = spec.policies.defaults.timeout_s
    for graph, node in ctx.nodes():
        if isinstance(node, OcrExtractNode | LexicalExtractNode | RagResolveNode):
            server = spec.mcp_servers.get(node.server)
            if server is None:
                continue
            needed, what = server.read_timeout_s, f"mcp_servers.{node.server}.read_timeout_s"
        elif isinstance(node, ScheduleNode):
            service = spec.http_services.get(node.service)
            if service is None:
                continue
            needed, what = service.timeout_s, f"http_services.{node.service}.timeout_s"
        else:
            continue
        timeout = default if node.timeout_s is None else node.timeout_s
        if timeout < needed:
            source = "timeout_s" if node.timeout_s is not None else "o timeout padrão"
            yield node_diagnostic(
                graph,
                node,
                Code.E_TIMEOUT_BUDGET,
                f"{source} do nó ({timeout:g}s) é menor que {what} ({needed:g}s)",
                field="timeout_s" if node.timeout_s is not None else None,
                hint=f"use timeout_s >= {needed:g}",
            )


def check(ctx: SemanticContext) -> list[Diagnostic]:
    return list(_budgets(ctx))
