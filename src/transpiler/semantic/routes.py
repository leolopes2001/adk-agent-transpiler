import json
from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic
from transpiler.registry import NodeKindSpec
from transpiler.semantic.context import (
    GraphView,
    IndexedEdge,
    SemanticContext,
    edge_diagnostic,
    kind_of,
    node_diagnostic,
)
from transpiler.spec.models import START, NodeSpec


def check(ctx: SemanticContext) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for graph in ctx.graphs:
        diagnostics.extend(_routed_edges(graph))
        diagnostics.extend(_uncovered_routes(graph))

    return diagnostics


def _routed_edges(graph: GraphView) -> Iterator[Diagnostic]:
    for edge in graph.known_edges():
        source = None if edge.source == START else graph.nodes[edge.source]
        spec = None if source is None else kind_of(source)
        if edge.route is None:
            yield from _missing_route(graph, edge, spec)
        else:
            yield from _invalid_route(graph, edge, source, spec)


def _missing_route(
    graph: GraphView, edge: IndexedEdge, spec: NodeKindSpec | None
) -> Iterator[Diagnostic]:
    if spec is not None and spec.is_router:
        yield edge_diagnostic(
            graph,
            edge,
            Code.E_MIXED_ROUTED_EDGES,
            f'"{edge.source}" é roteador; toda aresta de saída precisa de route',
            hint=_routes_hint(spec),
        )


def _invalid_route(
    graph: GraphView, edge: IndexedEdge, source: NodeSpec | None, spec: NodeKindSpec | None
) -> Iterator[Diagnostic]:
    if spec is None or not spec.is_router:
        what = "START" if source is None else f'"{edge.source}" ({source.kind})'
        yield edge_diagnostic(
            graph,
            edge,
            Code.E_ROUTE_ON_NON_ROUTER,
            f"{what} não emite rotas; remova route desta aresta",
            field="route",
        )
        return

    if edge.route not in spec.routes:
        yield edge_diagnostic(
            graph,
            edge,
            Code.E_UNKNOWN_ROUTE,
            f'a rota "{edge.route}" não é emitida por {spec.kind}',
            field="route",
            hint=_routes_hint(spec),
        )


def _uncovered_routes(graph: GraphView) -> Iterator[Diagnostic]:
    for node_id, node in graph.nodes.items():
        spec = kind_of(node)
        covered = {e.route for e in graph.outgoing(node_id)}
        for route in spec.routes:
            if route in covered:
                continue

            example = json.dumps(
                {"from": node_id, "to": "<nó>", "route": route}, ensure_ascii=False
            )
            yield node_diagnostic(
                graph,
                node,
                Code.E_ROUTE_NOT_COVERED,
                f'A rota "{route}" emitida por {spec.kind} não possui aresta de saída.',
                hint=f"adicione {example}",
            )


def _routes_hint(spec: NodeKindSpec) -> str:
    return f"rotas de {spec.kind}: {', '.join(spec.routes)}"
