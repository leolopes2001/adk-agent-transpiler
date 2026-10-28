from collections.abc import Iterator

from transpiler.errors import Code, Diagnostic
from transpiler.semantic.context import (
    GraphView,
    IndexedEdge,
    SemanticContext,
    edge_diagnostic,
    kind_of,
    node_diagnostic,
)
from transpiler.spec.models import START, JoinNodeSpec, LlmExtractNode, NodeSpec, SubworkflowNode


def check(ctx: SemanticContext) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for graph in ctx.graphs:
        diagnostics.extend(_duplicate_ids(graph))
        diagnostics.extend(_edge_endpoints(graph))
        diagnostics.extend(_start_and_reachability(graph))
        diagnostics.extend(_terminal_with_outgoing(graph))
        diagnostics.extend(_dead_ends(graph))
        diagnostics.extend(_duplicate_edges(graph))
        diagnostics.extend(_fan_in(graph))
        diagnostics.extend(_unconditional_cycles(graph))

    diagnostics.extend(_duplicate_adk_names(ctx))
    diagnostics.extend(_subworkflow_cycles(ctx))
    return diagnostics


def _duplicate_ids(graph: GraphView) -> Iterator[Diagnostic]:
    seen: set[str] = set()
    for position, node in enumerate(graph.spec.nodes):
        if node.id in seen:
            yield Diagnostic(
                code=Code.E_DUPLICATE_ID,
                message=f'id "{node.id}" repetido no {graph.label}',
                pointer=f"{graph.pointer}/nodes/{position}/id",
                node_id=node.id,
                kind=node.kind,
            )
        seen.add(node.id)


def _duplicate_adk_names(ctx: SemanticContext) -> Iterator[Diagnostic]:
    """Walks refs from the root: lowering adds names; a reused subworkflow emits nodes twice."""
    owners: dict[str, str] = {ctx.spec.agent.name: "/agent/name"}
    agents_seen: set[str] = set()

    def claim(name: str, pointer: str, node_id: str, kind: str) -> Iterator[Diagnostic]:
        if name not in owners:
            owners[name] = pointer
            return

        yield Diagnostic(
            code=Code.E_DUPLICATE_ID,
            message=f'nome ADK "{name}" repetido; já usado em {owners[name] or "/"}',
            pointer=pointer,
            node_id=node_id,
            kind=kind,
        )

    def walk(graph: GraphView, stack: tuple[str, ...]) -> Iterator[Diagnostic]:
        for node in graph.nodes.values():
            pointer = graph.node_pointer(node.id, "id")
            yield from claim(node.id, pointer, node.id, node.kind)
            if isinstance(node, LlmExtractNode):
                yield from claim(f"{node.id}_input", pointer, node.id, node.kind)
                if node.agent not in agents_seen:
                    agents_seen.add(node.agent)
                    agent_pointer = graph.node_pointer(node.id, "agent")
                    yield from claim(node.agent, agent_pointer, node.id, node.kind)
            if isinstance(node, SubworkflowNode) and node.ref not in stack:
                subgraph = ctx.subgraphs.get(node.ref)
                if subgraph is not None:
                    yield from walk(subgraph, (*stack, node.ref))

    yield from walk(ctx.root, ())


def _edge_endpoints(graph: GraphView) -> Iterator[Diagnostic]:
    for index, edge in enumerate(graph.spec.edges):
        indexed = IndexedEdge(index, edge)
        if edge.source != START and edge.source not in graph.nodes:
            yield edge_diagnostic(
                graph,
                indexed,
                Code.E_UNKNOWN_NODE,
                f'aresta sai de "{edge.source}", que não existe no {graph.label}',
                field="from",
            )
        if edge.target == START:
            yield edge_diagnostic(
                graph,
                indexed,
                Code.E_START_TARGET_INVALID,
                "START não pode ser destino de aresta",
                field="to",
            )
        elif edge.target not in graph.nodes:
            yield edge_diagnostic(
                graph,
                indexed,
                Code.E_UNKNOWN_NODE,
                f'aresta chega em "{edge.target}", que não existe no {graph.label}',
                field="to",
            )


def _start_and_reachability(graph: GraphView) -> Iterator[Diagnostic]:
    if not graph.outgoing(START):
        yield Diagnostic(
            code=Code.E_NO_START,
            message=f"o {graph.label} não tem aresta saindo de START",
            pointer=f"{graph.pointer}/edges",
            hint='adicione {"from": "START", "to": "<primeiro nó>"}',
        )
        return

    reached = graph.reachable_from_start()
    for node_id, node in graph.nodes.items():
        if node_id not in reached:
            yield node_diagnostic(
                graph, node, Code.E_UNREACHABLE, f'"{node_id}" não é alcançável a partir de START'
            )


def _terminal_with_outgoing(graph: GraphView) -> Iterator[Diagnostic]:
    for node_id, node in graph.nodes.items():
        if kind_of(node).terminal and graph.outgoing(node_id):
            yield node_diagnostic(
                graph,
                node,
                Code.E_TERMINAL_HAS_OUTGOING,
                f'"{node_id}" ({node.kind}) encerra o fluxo e não pode ter arestas de saída',
            )


def _dead_ends(graph: GraphView) -> Iterator[Diagnostic]:
    sinks = [graph.nodes[node_id] for node_id in graph.sinks()]
    if graph.is_root:
        for node in sinks:
            if not kind_of(node).terminal:
                yield node_diagnostic(
                    graph,
                    node,
                    Code.E_DEAD_END,
                    f'"{node.id}" não tem aresta de saída e {node.kind} não é terminal',
                    hint="ligue o nó a outro nó ou termine em present/abort",
                )
        return

    if len(sinks) > 1:
        names = ", ".join(f'"{node.id}"' for node in sinks)
        for node in sinks:
            yield node_diagnostic(
                graph,
                node,
                Code.E_DEAD_END,
                f"o {graph.label} deve ter um único nó final (sua saída); encontrados: {names}",
            )


def _duplicate_edges(graph: GraphView) -> Iterator[Diagnostic]:
    seen_edges: set[tuple[str, str, str | None]] = set()
    seen_routes: set[tuple[str, str]] = set()
    for index, edge in enumerate(graph.spec.edges):
        indexed = IndexedEdge(index, edge)
        key = (edge.source, edge.target, edge.route)
        if key in seen_edges:
            yield edge_diagnostic(graph, indexed, Code.E_DUPLICATE_EDGE, "aresta repetida")
        elif edge.route is not None and (edge.source, edge.route) in seen_routes:
            yield edge_diagnostic(
                graph,
                indexed,
                Code.E_DUPLICATE_EDGE,
                f'a rota "{edge.route}" de "{edge.source}" já tem destino',
                field="route",
            )
        seen_edges.add(key)
        if edge.route is not None:
            seen_routes.add((edge.source, edge.route))


def _fan_in(graph: GraphView) -> Iterator[Diagnostic]:
    for node in graph.nodes.values():
        if isinstance(node, JoinNodeSpec):
            yield from _join_inputs(graph, node)
        else:
            yield from _unjoined_fan_in(graph, node)


def _join_inputs(graph: GraphView, node: JoinNodeSpec) -> Iterator[Diagnostic]:
    incoming = graph.incoming(node.id)
    sources = {e.source for e in incoming}
    if len(sources) < 2:  # noqa: PLR2004
        yield node_diagnostic(
            graph,
            node,
            Code.E_JOIN_ARITY,
            f'join "{node.id}" precisa de ao menos 2 predecessores (tem {len(sources)})',
        )

    for edge in (e for e in incoming if e.route is not None):
        yield edge_diagnostic(
            graph,
            edge,
            Code.E_JOIN_ROUTED_INPUT,
            f'join "{node.id}" recebe a rota "{edge.route}" de "{edge.source}"; '
            "o join esperaria para sempre se essa rota não for tomada",
            field="route",
            hint="ligue ao join só arestas incondicionais (ramos que sempre executam)",
        )


def _unjoined_fan_in(graph: GraphView, node: NodeSpec) -> Iterator[Diagnostic]:
    unconditional = [e for e in graph.incoming(node.id) if e.route is None]
    if len(unconditional) >= 2:  # noqa: PLR2004
        names = ", ".join(f'"{e.source}"' for e in unconditional)
        yield node_diagnostic(
            graph,
            node,
            Code.E_FAN_IN_REQUIRES_JOIN,
            f'"{node.id}" recebe arestas incondicionais de {names}',
            hint="use um nó join para combinar as entradas",
        )


def _strongly_connected(graph: GraphView) -> list[list[str]]:
    """Tarjan over unconditional edges; only components that form a cycle are returned."""
    successors = {
        node_id: [
            e.target for e in graph.outgoing(node_id) if e.route is None and e.target in graph.nodes
        ]
        for node_id in graph.nodes
    }
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node_id: str) -> None:
        index[node_id] = low[node_id] = len(index)
        stack.append(node_id)
        on_stack.add(node_id)
        for target in successors[node_id]:
            if target not in index:
                visit(target)
                low[node_id] = min(low[node_id], low[target])
            elif target in on_stack:
                low[node_id] = min(low[node_id], index[target])

        if low[node_id] != index[node_id]:
            return

        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.discard(member)
            component.append(member)
            if member == node_id:
                break

        if len(component) > 1 or node_id in successors[node_id]:
            cycles.append(sorted(component, key=graph.positions.__getitem__))

    for node_id in graph.nodes:
        if node_id not in index:
            visit(node_id)

    return sorted(cycles, key=lambda c: graph.positions[c[0]])


def _unconditional_cycles(graph: GraphView) -> Iterator[Diagnostic]:
    for cycle in _strongly_connected(graph):
        first = graph.nodes[cycle[0]]
        members = " → ".join(cycle)
        yield node_diagnostic(
            graph,
            first,
            Code.E_UNCONDITIONAL_CYCLE,
            f"ciclo sem aresta condicional: {members}",
            hint="todo ciclo precisa passar por uma aresta com route",
        )


def _subworkflow_cycles(ctx: SemanticContext) -> Iterator[Diagnostic]:
    reported: set[tuple[str, str]] = set()

    def walk(graph: GraphView, stack: tuple[str, ...]) -> Iterator[Diagnostic]:
        for node in graph.nodes.values():
            if not isinstance(node, SubworkflowNode) or node.ref not in ctx.subgraphs:
                continue

            if node.ref not in stack:
                yield from walk(ctx.subgraphs[node.ref], (*stack, node.ref))
                continue

            if (graph.key, node.id) in reported:
                continue

            reported.add((graph.key, node.id))
            chain = " → ".join((*stack[stack.index(node.ref) :], node.ref))
            yield node_diagnostic(
                graph,
                node,
                Code.E_SUBWORKFLOW_CYCLE,
                f"subworkflow referencia a si mesmo: {chain}",
                field="ref",
            )

    yield from walk(ctx.root, ())
    for name, graph in ctx.subgraphs.items():
        yield from walk(graph, (name,))
