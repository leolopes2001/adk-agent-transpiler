"""Rewrites IR kinds into ADK primitives; each Python variable name is also the ADK node name."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from transpiler.ir.model import (
    GraphIR,
    HttpServiceIR,
    LlmAgentIR,
    McpServerIR,
    NodeIR,
    ProgramIR,
    RetryIR,
)
from transpiler.registry import NodeKind, PortType, kind_spec
from transpiler.spec.models import START

_LLM_SOURCE = "llm"
_LEXICAL_SOURCE = "lexical"

type PyLiteral = str | int | float | bool | tuple[PyLiteral, ...] | Mapping[str, PyLiteral] | None


@dataclass(frozen=True, slots=True)
class AdkNode:
    var: str
    primitive: Literal["function", "agent", "join", "workflow"]
    factory: str | None = None
    kwargs: Mapping[str, PyLiteral] = field(default_factory=lambda: MappingProxyType({}))
    needs_deps: bool = False
    retry: RetryIR | None = None
    timeout_s: float | None = None
    agent: LlmAgentIR | None = None
    subgraph: AdkGraph | None = None


@dataclass(frozen=True, slots=True)
class AdkEdge:
    source: str
    target: str | None = None
    routes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AdkGraph:
    name: str
    nodes: tuple[AdkNode, ...]
    """Direct members, in stable topological order."""
    edges: tuple[AdkEdge, ...]


@dataclass(frozen=True, slots=True)
class AdkProgram:
    agent_name: str
    description: str
    version: str
    root: AdkGraph
    servers: tuple[McpServerIR, ...]
    services: tuple[HttpServiceIR, ...]
    needs_resumability: bool
    spec_sha256: str
    ir_digest: str

    def leaf_nodes(self) -> list[AdkNode]:
        def walk(graph: AdkGraph) -> Iterator[AdkNode]:
            for node in graph.nodes:
                if node.subgraph is not None:
                    yield from walk(node.subgraph)
                else:
                    yield node

        return list(walk(self.root))

    def subworkflows(self) -> list[AdkNode]:
        """Innermost first, so each is defined before its parent uses it."""

        def walk(graph: AdkGraph) -> Iterator[AdkNode]:
            for node in graph.nodes:
                if node.subgraph is not None:
                    yield from walk(node.subgraph)
                    yield node

        return list(walk(self.root))

    def node_names(self) -> list[str]:
        return sorted(node.var for node in (*self.leaf_nodes(), *self.subworkflows()))

    def uses(self, primitive: str) -> bool:
        return any(node.primitive == primitive for node in self.leaf_nodes())

    @property
    def uses_retry_config(self) -> bool:
        return any(n.retry is not None for n in self.leaf_nodes() if n.primitive == "function")


@dataclass
class _EdgeBuilder:
    edges: list[AdkEdge] = field(default_factory=list)
    router_slots: dict[str, int] = field(default_factory=dict)
    router_routes: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def add_plain(self, source: str, target: str) -> None:
        self.edges.append(AdkEdge(source=source, target=target))

    def add_route(self, source: str, route: str, target: str) -> None:
        if source not in self.router_slots:
            self.router_slots[source] = len(self.edges)
            self.router_routes[source] = []
            self.edges.append(AdkEdge(source=source))

        self.router_routes[source].append((route, target))

    def build(self) -> tuple[AdkEdge, ...]:
        for source, slot in self.router_slots.items():
            self.edges[slot] = AdkEdge(source=source, routes=tuple(self.router_routes[source]))

        return tuple(self.edges)


class _Lowering:
    def lower_graph(self, graph: GraphIR, name: str) -> AdkGraph:
        entries: dict[str, str] = {}
        exits: dict[str, str] = {}
        nodes: list[AdkNode] = []
        for node in _topological_order(graph):
            lowered = self.lower_node(graph, node)
            nodes.extend(lowered)
            entries[node.id], exits[node.id] = lowered[0].var, lowered[-1].var

        return AdkGraph(
            name=name, nodes=tuple(nodes), edges=self.lower_edges(graph, entries, exits)
        )

    def lower_node(self, graph: GraphIR, node: NodeIR) -> list[AdkNode]:
        if node.agent is not None:
            return _llm_extract_nodes(node, node.agent)

        if node.kind is NodeKind.JOIN:
            return [AdkNode(var=node.id, primitive="join")]

        if node.subgraph is not None:
            subgraph = self.lower_graph(node.subgraph, node.id)
            return [AdkNode(var=node.id, primitive="workflow", subgraph=subgraph)]

        return [_function_node(graph, node)]

    def lower_edges(
        self, graph: GraphIR, entries: Mapping[str, str], exits: Mapping[str, str]
    ) -> tuple[AdkEdge, ...]:
        builder = _EdgeBuilder()
        expanded: set[str] = set()
        for edge in graph.edges:
            source = START if edge.source == START else exits[edge.source]
            target = entries[edge.target]
            if edge.route is None:
                builder.add_plain(source, target)
            else:
                builder.add_route(source, edge.route, target)

            if target != exits[edge.target] and edge.target not in expanded:
                expanded.add(edge.target)
                builder.add_plain(target, exits[edge.target])

        return builder.build()


def lower(program: ProgramIR) -> AdkProgram:
    return AdkProgram(
        agent_name=program.agent_name,
        description=program.description,
        version=program.version,
        root=_Lowering().lower_graph(program.root, program.agent_name),
        servers=program.servers,
        services=program.services,
        needs_resumability=program.needs_resumability,
        spec_sha256=program.spec_sha256,
        ir_digest=program.digest(),
    )


def _topological_order(graph: GraphIR) -> list[NodeIR]:
    """Kahn's algorithm; ties and conditional cycles fall back to spec order."""
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    pending = _in_degrees(graph)
    remaining = dict(position)
    order: list[NodeIR] = []
    while remaining:
        ready = [node_id for node_id in remaining if pending[node_id] == 0]
        chosen = min(ready or remaining, key=position.__getitem__)
        del remaining[chosen]
        order.append(graph.nodes[position[chosen]])
        for edge in graph.edges:
            if edge.source == chosen and edge.target in remaining:
                pending[edge.target] -= 1

    return order


def _in_degrees(graph: GraphIR) -> dict[str, int]:
    degrees = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        if edge.source != START:
            degrees[edge.target] += 1

    return degrees


def _llm_extract_nodes(node: NodeIR, agent: LlmAgentIR) -> list[AdkNode]:
    return [
        AdkNode(var=f"{node.id}_input", primitive="function", factory=kind_spec(node.kind).factory),
        AdkNode(
            var=agent.key,
            primitive="agent",
            agent=agent,
            retry=node.retry,
            timeout_s=node.timeout_s,
        ),
    ]


def _function_node(graph: GraphIR, node: NodeIR) -> AdkNode:
    return AdkNode(
        var=node.id,
        primitive="function",
        factory=kind_spec(node.kind).factory,
        kwargs=MappingProxyType(_function_kwargs(graph, node)),
        needs_deps=node.server is not None or node.service is not None,
        retry=node.retry,
        timeout_s=node.timeout_s,
    )


def _function_kwargs(graph: GraphIR, node: NodeIR) -> dict[str, PyLiteral]:
    kwargs: dict[str, PyLiteral] = {}
    if node.server is not None:
        kwargs["server"] = node.server.key
        kwargs["tool"] = node.tool
    if node.service is not None:
        kwargs["service"] = node.service.key
    if node.kind is NodeKind.RECONCILE:
        kwargs["sources"] = _reconcile_sources(graph, node)
    kwargs.update(node.params)
    return kwargs


def _reconcile_sources(graph: GraphIR, node: NodeIR) -> Mapping[str, PyLiteral]:
    [join_id] = [e.source for e in graph.edges if e.target == node.id]
    sources: dict[str, str] = {}
    for edge in graph.edges:
        if edge.target != join_id:
            continue

        predecessor = graph.node(edge.source)
        is_llm = _output_type(predecessor) is PortType.LLM_MENTIONS
        sources[_LLM_SOURCE if is_llm else _LEXICAL_SOURCE] = _exit_var(predecessor)

    return MappingProxyType(
        {_LLM_SOURCE: sources[_LLM_SOURCE], _LEXICAL_SOURCE: sources[_LEXICAL_SOURCE]}
    )


def _exit_var(node: NodeIR) -> str:
    return node.id if node.agent is None else node.agent.key


def _output_type(node: NodeIR) -> PortType | None:
    if node.subgraph is not None:
        sources = {e.source for e in node.subgraph.edges}
        [sink] = [n for n in node.subgraph.nodes if n.id not in sources]
        return _output_type(sink)

    return kind_spec(node.kind).output_type
