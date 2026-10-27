"""Lowering: rewrites IR node kinds into the ADK primitives the template emits.

After this pass every node is a FunctionNode (built by a runtime factory), an
Agent, a JoinNode or a nested Workflow, and every edge refers to Python
variable names, which are also the ADK node names. Two kinds need more than a
one-to-one mapping:

- `llm_extract X` with agent `A` becomes `X_input` (a FunctionNode that builds
  the model prompt) followed by the Agent `A`; edges into X go to `X_input` and
  edges out of X leave from `A`.
- `reconcile` gets a computed `sources` kwarg naming the join's predecessors,
  so the runtime can pick the LLM and lexical results out of the join output.
"""

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

type PyLiteral = str | int | float | bool | tuple[PyLiteral, ...] | Mapping[str, PyLiteral] | None


@dataclass(frozen=True, slots=True)
class AdkNode:
    var: str
    """Python variable name, equal to the ADK node name."""
    primitive: Literal["function", "agent", "join", "workflow"]
    factory: str | None = None
    """`agent_runtime.nodes` factory, for function nodes."""
    kwargs: Mapping[str, PyLiteral] = field(default_factory=lambda: MappingProxyType({}))
    needs_deps: bool = False
    retry: RetryIR | None = None
    timeout_s: float | None = None
    agent: LlmAgentIR | None = None
    subgraph: AdkGraph | None = None


@dataclass(frozen=True, slots=True)
class AdkEdge:
    """`(source, target)` for a plain edge; `(source, {route: target})` for a router."""

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
        """Function, agent and join nodes in emission order (nested graphs expanded in place)."""

        def walk(graph: AdkGraph) -> Iterator[AdkNode]:
            for node in graph.nodes:
                if node.subgraph is not None:
                    yield from walk(node.subgraph)
                else:
                    yield node

        return list(walk(self.root))

    def subworkflows(self) -> list[AdkNode]:
        """Nested workflow nodes, innermost first, so each is defined before its parent uses it."""

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


def _topological(graph: GraphIR) -> list[NodeIR]:
    """Kahn's algorithm, breaking ties (and conditional cycles) by spec order."""
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    pending = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        if edge.source != START:
            pending[edge.target] += 1
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


def _output_type(node: NodeIR) -> PortType | None:
    if node.subgraph is not None:
        sources = {e.source for e in node.subgraph.edges}
        [sink] = [n for n in node.subgraph.nodes if n.id not in sources]
        return _output_type(sink)
    return kind_spec(node.kind).output_type


class _Lowering:
    def graph(self, graph: GraphIR, name: str) -> AdkGraph:
        entry: dict[str, str] = {}
        exit_: dict[str, str] = {}
        nodes: list[AdkNode] = []
        for node in _topological(graph):
            lowered = self.node(graph, node)
            nodes.extend(lowered)
            entry[node.id], exit_[node.id] = lowered[0].var, lowered[-1].var
        return AdkGraph(name=name, nodes=tuple(nodes), edges=self.edges(graph, entry, exit_))

    def node(self, graph: GraphIR, node: NodeIR) -> list[AdkNode]:
        spec = kind_spec(node.kind)
        if node.agent is not None:  # llm_extract
            return [
                AdkNode(var=f"{node.id}_input", primitive="function", factory=spec.factory),
                AdkNode(
                    var=node.agent.key,
                    primitive="agent",
                    agent=node.agent,
                    retry=node.retry,
                    timeout_s=node.timeout_s,
                ),
            ]
        if node.kind is NodeKind.JOIN:
            return [AdkNode(var=node.id, primitive="join")]
        if node.subgraph is not None:
            return [
                AdkNode(
                    var=node.id, primitive="workflow", subgraph=self.graph(node.subgraph, node.id)
                )
            ]

        kwargs: dict[str, PyLiteral] = {}
        if node.server is not None:
            kwargs["server"] = node.server.key
            kwargs["tool"] = node.tool
        if node.service is not None:
            kwargs["service"] = node.service.key
        if node.kind is NodeKind.RECONCILE:
            kwargs["sources"] = self.reconcile_sources(graph, node)
        kwargs.update(node.params)
        return [
            AdkNode(
                var=node.id,
                primitive="function",
                factory=spec.factory,
                kwargs=MappingProxyType(kwargs),
                needs_deps=node.server is not None or node.service is not None,
                retry=node.retry,
                timeout_s=node.timeout_s,
            )
        ]

    def reconcile_sources(self, graph: GraphIR, node: NodeIR) -> Mapping[str, PyLiteral]:
        [join_id] = [e.source for e in graph.edges if e.target == node.id]
        sources: dict[str, str] = {}
        for edge in graph.edges:
            if edge.target != join_id:
                continue
            predecessor = graph.node(edge.source)
            role = "llm" if _output_type(predecessor) is PortType.LLM_MENTIONS else "lexical"
            # An llm_extract node is lowered to its Agent, which is what the join sees.
            agent = predecessor.agent
            sources[role] = predecessor.id if agent is None else agent.key
        return MappingProxyType({"llm": sources["llm"], "lexical": sources["lexical"]})

    def edges(
        self, graph: GraphIR, entry: Mapping[str, str], exit_: Mapping[str, str]
    ) -> tuple[AdkEdge, ...]:
        def source_var(source: str) -> str:
            return START if source == START else exit_[source]

        edges: list[AdkEdge] = []
        router_slots: dict[str, int] = {}
        router_routes: dict[str, list[tuple[str, str]]] = {}
        internal_done: set[str] = set()
        for edge in graph.edges:
            target = entry[edge.target]
            if edge.route is None:
                edges.append(AdkEdge(source=source_var(edge.source), target=target))
            else:
                if edge.source not in router_slots:
                    router_slots[edge.source] = len(edges)
                    router_routes[edge.source] = []
                    edges.append(AdkEdge(source=source_var(edge.source)))
                router_routes[edge.source].append((edge.route, target))
            # The edge inside a lowered llm_extract follows the first edge into it.
            if entry[edge.target] != exit_[edge.target] and edge.target not in internal_done:
                internal_done.add(edge.target)
                edges.append(AdkEdge(source=entry[edge.target], target=exit_[edge.target]))
        for source, slot in router_slots.items():
            edges[slot] = AdkEdge(source=source_var(source), routes=tuple(router_routes[source]))
        return tuple(edges)


def lower(program: ProgramIR) -> AdkProgram:
    return AdkProgram(
        agent_name=program.agent_name,
        description=program.description,
        version=program.version,
        root=_Lowering().graph(program.root, program.agent_name),
        servers=program.servers,
        services=program.services,
        needs_resumability=program.needs_resumability,
        spec_sha256=program.spec_sha256,
        ir_digest=program.digest(),
    )
