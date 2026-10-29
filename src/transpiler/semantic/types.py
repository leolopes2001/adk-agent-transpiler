from collections.abc import Iterator
from dataclasses import dataclass, field

from transpiler.errors import Code, Diagnostic
from transpiler.registry import PortType
from transpiler.semantic.context import (
    GraphView,
    IndexedEdge,
    SemanticContext,
    edge_diagnostic,
    kind_of,
)
from transpiler.spec.models import (
    START,
    ConfidenceGateNode,
    JoinNodeSpec,
    NodeSpec,
    RagResolveNode,
    SubworkflowNode,
)

# The gate compares each mention's two best candidates; with one, a tie looks clear (audit F03).
GATE_MIN_CANDIDATES = 2

type Types = frozenset[PortType]


@dataclass
class _TypeChecker:
    ctx: SemanticContext
    start_types: dict[str, set[PortType]] = field(default_factory=dict)

    def output_types(
        self, graph: GraphView, node: NodeSpec, stack: tuple[str, ...]
    ) -> Types | None:
        if not isinstance(node, SubworkflowNode):
            return kind_of(node).output_types()

        subgraph = self.ctx.subgraphs.get(node.ref)
        if subgraph is None or node.ref in stack:
            return None

        sinks = subgraph.sinks()
        if len(sinks) != 1:
            return None

        return self.output_types(subgraph, subgraph.nodes[sinks[0]], (*stack, node.ref))

    def produced_types(self, graph: GraphView, edge: IndexedEdge) -> Types | None:
        if edge.source == START:
            return self._start_types(graph)

        source = graph.nodes[edge.source]
        spec = kind_of(source)
        if spec.terminal:
            return None

        if spec.is_router:
            route_type = spec.routes.get(edge.route) if edge.route else None
            return None if route_type is None else frozenset({route_type})

        return self.output_types(graph, source, (graph.key,))

    def join_components(self, graph: GraphView, join_id: str) -> tuple[PortType, ...] | None:
        components: list[PortType] = []
        for edge in graph.incoming(join_id):
            if edge.source != START and edge.source not in graph.nodes:
                continue

            produced = self.produced_types(graph, edge)
            if produced is None or len(produced) != 1:
                return None

            components.extend(produced)

        return tuple(sorted(components))

    def check_graph(self, graph: GraphView) -> Iterator[Diagnostic]:
        for edge in graph.known_edges():
            produced = self.produced_types(graph, edge)
            if produced is None:
                continue

            target = graph.nodes[edge.target]
            if isinstance(target, SubworkflowNode):
                self.start_types.setdefault(target.ref, set()).update(produced)
                continue

            accepted = kind_of(target).input_types
            if accepted and not produced <= accepted:
                via = f' pela rota "{edge.route}"' if edge.route else ""
                yield edge_diagnostic(
                    graph,
                    edge,
                    Code.E_TYPE_MISMATCH,
                    f'"{edge.source}" produz {_type_names(produced)}{via}, mas "{target.id}" '
                    f"({target.kind}) espera {_type_names(accepted)}",
                )
                continue

            yield from self._check_join_inputs(graph, edge, target)
            yield from self._check_gate_candidates(graph, edge, target)

    def _start_types(self, graph: GraphView) -> Types | None:
        if graph.is_root:
            return frozenset({PortType.RUN_INPUT})

        known = self.start_types.get(graph.key)
        return frozenset(known) if known else None

    def _check_gate_candidates(
        self, graph: GraphView, edge: IndexedEdge, target: NodeSpec
    ) -> Iterator[Diagnostic]:
        source = graph.nodes.get(edge.source)
        if not isinstance(target, ConfidenceGateNode) or not isinstance(source, RagResolveNode):
            return

        if source.params.top_k < GATE_MIN_CANDIDATES:
            yield Diagnostic(
                code=Code.E_GATE_CANDIDATES,
                message=(
                    f'"{source.id}" alimenta o confidence_gate "{target.id}" com top_k='
                    f"{source.params.top_k}; o gate compara os dois melhores candidatos de "
                    "cada exame e, com um só, não detecta ambiguidade"
                ),
                pointer=graph.node_pointer(source.id, "params", "top_k"),
                hint=f"use top_k >= {GATE_MIN_CANDIDATES}",
                node_id=source.id,
                kind=source.kind,
            )

    def _check_join_inputs(
        self, graph: GraphView, edge: IndexedEdge, target: NodeSpec
    ) -> Iterator[Diagnostic]:
        """Lowering names the join's inputs, which it cannot do across subworkflows (audit F05)."""
        required = kind_of(target).join_inputs
        if not required:
            return

        source = graph.nodes.get(edge.source)
        if not isinstance(source, JoinNodeSpec):
            what = "START" if source is None else f'"{source.id}" ({source.kind})'
            yield edge_diagnostic(
                graph,
                edge,
                Code.E_RECONCILE_WITHOUT_JOIN,
                f'"{target.id}" ({target.kind}) precisa receber diretamente um join do mesmo '
                f"{graph.label}; recebe de {what}",
                hint="ponha os extratores e o join no mesmo grafo que o reconcile",
            )
            return

        components = self.join_components(graph, source.id)
        if components is not None and components != required:
            yield edge_diagnostic(
                graph,
                edge,
                Code.E_TYPE_MISMATCH,
                f'"{target.id}" ({target.kind}) precisa de um join de exatamente '
                f'{_type_names(required)}; "{source.id}" combina {_type_names(components)}',
            )


def check(ctx: SemanticContext) -> list[Diagnostic]:
    checker = _TypeChecker(ctx)
    return [d for graph in _graphs_parents_first(ctx) for d in checker.check_graph(graph)]


def _graphs_parents_first(ctx: SemanticContext) -> Iterator[GraphView]:
    pending, seen = [ctx.root], {""}
    while pending:
        graph = pending.pop(0)
        yield graph
        for node in graph.nodes.values():
            if not isinstance(node, SubworkflowNode) or node.ref in seen:
                continue

            subgraph = ctx.subgraphs.get(node.ref)
            if subgraph is not None:
                seen.add(node.ref)
                pending.append(subgraph)


def _type_names(types: Types | tuple[PortType, ...]) -> str:
    return " | ".join(sorted(types)) or "nada"
