"""Indexed, read-only view of a spec for the semantic checks.

Checks must tolerate broken graphs (unknown nodes, duplicated ids, cycles)
because all of them run and report together; the views below therefore only
index what is well defined and leave the judging to each check.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from transpiler.errors import Code, Diagnostic, json_pointer
from transpiler.registry import NodeKindSpec, kind_spec
from transpiler.spec.models import START, AgentSpec, EdgeSpec, GraphSpec, NodeSpec


@dataclass(frozen=True, slots=True)
class IndexedEdge:
    index: int
    edge: EdgeSpec

    @property
    def source(self) -> str:
        return self.edge.source

    @property
    def target(self) -> str:
        return self.edge.target

    @property
    def route(self) -> str | None:
        return self.edge.route


@dataclass(frozen=True, slots=True)
class GraphView:
    key: str
    """`""` for the root workflow, the subworkflow name otherwise."""
    base: tuple[str, ...]
    spec: GraphSpec
    nodes: Mapping[str, NodeSpec]
    """First node with each id, in spec order."""
    positions: Mapping[str, int]
    out_edges: Mapping[str, tuple[IndexedEdge, ...]] = field(repr=False)
    in_edges: Mapping[str, tuple[IndexedEdge, ...]] = field(repr=False)

    @classmethod
    def build(cls, key: str, base: tuple[str, ...], spec: GraphSpec) -> GraphView:
        nodes: dict[str, NodeSpec] = {}
        positions: dict[str, int] = {}
        for position, node in enumerate(spec.nodes):
            if node.id not in nodes:
                nodes[node.id] = node
                positions[node.id] = position
        out_edges: defaultdict[str, list[IndexedEdge]] = defaultdict(list)
        in_edges: defaultdict[str, list[IndexedEdge]] = defaultdict(list)
        for index, edge in enumerate(spec.edges):
            indexed = IndexedEdge(index, edge)
            out_edges[edge.source].append(indexed)
            in_edges[edge.target].append(indexed)
        return cls(
            key=key,
            base=base,
            spec=spec,
            nodes=nodes,
            positions=positions,
            out_edges={k: tuple(v) for k, v in out_edges.items()},
            in_edges={k: tuple(v) for k, v in in_edges.items()},
        )

    @property
    def is_root(self) -> bool:
        return self.key == ""

    @property
    def label(self) -> str:
        return "workflow" if self.is_root else f'subworkflow "{self.key}"'

    @property
    def pointer(self) -> str:
        return json_pointer(self.base)

    def node_pointer(self, node_id: str, *fields: str) -> str:
        return json_pointer((*self.base, "nodes", self.positions[node_id], *fields))

    def edge_pointer(self, edge: IndexedEdge, *fields: str) -> str:
        return json_pointer((*self.base, "edges", edge.index, *fields))

    def outgoing(self, node_id: str) -> tuple[IndexedEdge, ...]:
        return self.out_edges.get(node_id, ())

    def incoming(self, node_id: str) -> tuple[IndexedEdge, ...]:
        return self.in_edges.get(node_id, ())

    def known_edges(self) -> Iterator[IndexedEdge]:
        """Edges whose endpoints exist (START counts as a source only)."""
        for index, edge in enumerate(self.spec.edges):
            source_ok = edge.source == START or edge.source in self.nodes
            if source_ok and edge.target in self.nodes:
                yield IndexedEdge(index, edge)

    def successors(self, node_id: str) -> list[str]:
        return [e.target for e in self.outgoing(node_id) if e.target in self.nodes]

    def reachable_from_start(self, removed: frozenset[str] = frozenset()) -> set[str]:
        seen: set[str] = set()
        stack = [t for t in self.successors(START) if t not in removed]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(t for t in self.successors(node_id) if t not in removed)
        return seen

    def sinks(self) -> list[str]:
        return [node_id for node_id in self.nodes if not self.outgoing(node_id)]


@dataclass(frozen=True, slots=True)
class SemanticContext:
    spec: AgentSpec
    spec_dir: Path
    root: GraphView
    subgraphs: Mapping[str, GraphView]

    @classmethod
    def build(cls, spec: AgentSpec, spec_dir: Path) -> SemanticContext:
        return cls(
            spec=spec,
            spec_dir=spec_dir,
            root=GraphView.build("", ("workflow",), spec.workflow),
            subgraphs={
                name: GraphView.build(name, ("subworkflows", name), graph)
                for name, graph in spec.subworkflows.items()
            },
        )

    @property
    def graphs(self) -> tuple[GraphView, ...]:
        return (self.root, *self.subgraphs.values())

    def nodes(self) -> Iterator[tuple[GraphView, NodeSpec]]:
        for graph in self.graphs:
            for node in graph.nodes.values():
                yield graph, node


def kind_of(node: NodeSpec) -> NodeKindSpec:
    return kind_spec(node.kind)


def node_diagnostic(  # noqa: PLR0913
    graph: GraphView,
    node: NodeSpec,
    code: Code,
    message: str,
    *,
    field: str | None = None,
    hint: str | None = None,
) -> Diagnostic:
    fields = () if field is None else (field,)
    return Diagnostic(
        code=code,
        message=message,
        pointer=graph.node_pointer(node.id, *fields),
        hint=hint,
        node_id=node.id,
        kind=node.kind,
    )


def edge_diagnostic(  # noqa: PLR0913
    graph: GraphView,
    edge: IndexedEdge,
    code: Code,
    message: str,
    *,
    field: str | None = None,
    hint: str | None = None,
) -> Diagnostic:
    fields = () if field is None else (field,)
    node = graph.nodes.get(edge.source)
    return Diagnostic(
        code=code,
        message=message,
        pointer=graph.edge_pointer(edge, *fields),
        hint=hint,
        node_id=edge.source if node else None,
        kind=node.kind if node else None,
    )
