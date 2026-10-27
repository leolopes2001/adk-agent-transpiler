"""Port type compatibility along every edge.

A subworkflow is typed by its contents: its input is whatever the parent
sends into it and its output is the output of its single final node. A join
produces a `JoinResult` whose components are the types of its inputs, which
lets `reconcile` require exactly one LLM and one lexical source.
"""

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
from transpiler.spec.models import START, JoinNodeSpec, NodeSpec, SubworkflowNode

type Types = frozenset[PortType]


def _names(types: Types | tuple[PortType, ...]) -> str:
    return " | ".join(sorted(types)) or "nada"


@dataclass
class _Typer:
    ctx: SemanticContext
    start_types: dict[str, set[PortType]] = field(default_factory=dict)

    def node_output(self, graph: GraphView, node: NodeSpec, stack: tuple[str, ...]) -> Types | None:
        """Every type the node may emit; None when it cannot be determined."""
        if isinstance(node, SubworkflowNode):
            subgraph = self.ctx.subgraphs.get(node.ref)
            if subgraph is None or node.ref in stack:
                return None
            sinks = subgraph.sinks()
            if len(sinks) != 1:
                return None
            return self.node_output(subgraph, subgraph.nodes[sinks[0]], (*stack, node.ref))
        return kind_of(node).output_types()

    def produced(self, graph: GraphView, edge: IndexedEdge) -> Types | None:
        if edge.source == START:
            if graph.is_root:
                return frozenset({PortType.RUN_INPUT})
            known = self.start_types.get(graph.key)
            return frozenset(known) if known else None
        source = graph.nodes[edge.source]
        spec = kind_of(source)
        if spec.terminal:
            return None
        if spec.is_router:
            route_type = spec.routes.get(edge.route) if edge.route else None
            return None if route_type is None else frozenset({route_type})
        return self.node_output(graph, source, (graph.key,))

    def join_components(self, graph: GraphView, join_id: str) -> tuple[PortType, ...] | None:
        components: list[PortType] = []
        for edge in graph.incoming(join_id):
            if edge.source != START and edge.source not in graph.nodes:
                continue
            produced = self.produced(graph, edge)
            if produced is None or len(produced) != 1:
                return None
            components.extend(produced)
        return tuple(sorted(components))

    def check_graph(self, graph: GraphView) -> Iterator[Diagnostic]:
        for edge in graph.known_edges():
            produced = self.produced(graph, edge)
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
                    f'"{edge.source}" produz {_names(produced)}{via}, mas "{target.id}" '
                    f"({target.kind}) espera {_names(accepted)}",
                )
                continue
            yield from self._check_join_inputs(graph, edge, target)

    def _check_join_inputs(
        self, graph: GraphView, edge: IndexedEdge, target: NodeSpec
    ) -> Iterator[Diagnostic]:
        required = kind_of(target).join_inputs
        source = graph.nodes.get(edge.source)
        if not required or not isinstance(source, JoinNodeSpec):
            return
        components = self.join_components(graph, source.id)
        if components is not None and components != required:
            yield edge_diagnostic(
                graph,
                edge,
                Code.E_TYPE_MISMATCH,
                f'"{target.id}" ({target.kind}) precisa de um join de exatamente '
                f'{_names(required)}; "{source.id}" combina {_names(components)}',
            )


def check(ctx: SemanticContext) -> list[Diagnostic]:
    typer = _Typer(ctx)
    diagnostics: list[Diagnostic] = []
    # Parents first, so a subworkflow knows its input type before it is checked.
    pending, seen = [ctx.root], {""}
    while pending:
        graph = pending.pop(0)
        diagnostics.extend(typer.check_graph(graph))
        for node in graph.nodes.values():
            if isinstance(node, SubworkflowNode) and node.ref not in seen:
                subgraph = ctx.subgraphs.get(node.ref)
                if subgraph is not None:
                    seen.add(node.ref)
                    pending.append(subgraph)
    return diagnostics
