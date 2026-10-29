"""Guards are checked by removal on the inlined program, so a nested graph cannot hide a path."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from transpiler.errors import Code, Diagnostic, json_pointer
from transpiler.registry import NodeKind
from transpiler.semantic.context import GraphView, SemanticContext, node_diagnostic
from transpiler.spec.models import START, LlmExtractNode, NodeSpec, SubworkflowNode

_DETERMINISTIC_TOOLS_HINT = "chamadas MCP pertencem a nós determinísticos (ADR-0004)"


@dataclass
class _FlatGraph:
    """Node keys are `parent/child` paths through the inlined subworkflows."""

    owners: dict[str, tuple[GraphView, NodeSpec]] = field(default_factory=dict)
    successors: dict[str, set[str]] = field(default_factory=dict)

    def add_edge(self, source: str, target: str) -> None:
        self.successors.setdefault(source, set()).add(target)

    def reachable(self, removed_kinds: frozenset[str]) -> set[str]:
        def allowed(key: str) -> bool:
            return self.owners[key][1].kind not in removed_kinds

        seen: set[str] = set()
        stack = [k for k in self.successors.get(START, ()) if allowed(k)]
        while stack:
            key = stack.pop()
            if key in seen:
                continue

            seen.add(key)
            stack.extend(k for k in self.successors.get(key, ()) if allowed(k))

        return seen


def check(ctx: SemanticContext) -> list[Diagnostic]:
    return [
        *_guard_and_gate(ctx),
        *_agent_tools(ctx),
        *_llm_input_trust(ctx),
        *_pii_policy(ctx),
    ]


def _flatten(ctx: SemanticContext) -> _FlatGraph:
    flat = _FlatGraph()

    def expand(
        graph: GraphView, prefix: str, stack: tuple[str, ...]
    ) -> tuple[list[str], list[str]]:
        """Returns the keys the graph is entered through and the keys it is left from."""
        entries: dict[str, list[str]] = {}
        exits: dict[str, list[str]] = {}
        for node_id, node in graph.nodes.items():
            key = f"{prefix}{node_id}"
            subgraph = ctx.subgraphs.get(node.ref) if isinstance(node, SubworkflowNode) else None
            if isinstance(node, SubworkflowNode) and subgraph and node.ref not in stack:
                entries[node_id], exits[node_id] = expand(subgraph, f"{key}/", (*stack, node.ref))
            else:
                flat.owners[key] = (graph, node)
                entries[node_id] = exits[node_id] = [key]

        start: list[str] = []
        for edge in graph.known_edges():
            targets = entries[edge.target]
            if edge.source == START:
                start.extend(targets)
                continue

            for source in exits[edge.source]:
                for target in targets:
                    flat.add_edge(source, target)

        sinks = [key for node_id in graph.sinks() for key in exits[node_id]]
        return start, sinks

    start, _ = expand(ctx.root, "", ())
    for target in start:
        flat.add_edge(START, target)

    return flat


def _unguarded(
    flat: _FlatGraph,
    *,
    protected: NodeKind,
    guardian: NodeKind,
    code: Code,
    message: str,
) -> Iterator[Diagnostic]:
    reported: set[tuple[str, str]] = set()
    for key in sorted(flat.reachable(frozenset({guardian.value}))):
        graph, node = flat.owners[key]
        if node.kind == protected and (graph.key, node.id) not in reported:
            reported.add((graph.key, node.id))
            yield node_diagnostic(graph, node, code, message.format(node=node.id))


def _guard_and_gate(ctx: SemanticContext) -> Iterator[Diagnostic]:
    flat = _flatten(ctx)
    yield from _unguarded(
        flat,
        protected=NodeKind.LLM_EXTRACT,
        guardian=NodeKind.UNTRUSTED_TEXT_GUARD,
        code=Code.E_LLM_NOT_GUARDED,
        message='existe caminho de START até "{node}" que não passa por untrusted_text_guard',
    )
    yield from _unguarded(
        flat,
        protected=NodeKind.SCHEDULE,
        guardian=NodeKind.CONFIDENCE_GATE,
        code=Code.E_SCHEDULE_NOT_GATED,
        message='existe caminho de START até "{node}" que não passa por confidence_gate',
    )


def _agent_tools(ctx: SemanticContext) -> Iterator[Diagnostic]:
    for name, agent in ctx.spec.llm_agents.items():
        if not agent.tools:
            continue

        pointer = json_pointer(("llm_agents", name, "tools"))
        if agent.input_trust == "untrusted":
            yield Diagnostic(
                code=Code.E_UNTRUSTED_AGENT_HAS_TOOLS,
                message=f'o agente "{name}" recebe dado não confiável e não pode ter tools',
                pointer=pointer,
                hint=_DETERMINISTIC_TOOLS_HINT,
            )
        else:
            yield Diagnostic(
                code=Code.E_AGENT_TOOLS_UNSUPPORTED,
                message=f'tools em agentes não são suportadas pelo gerador (agente "{name}")',
                pointer=pointer,
                hint=_DETERMINISTIC_TOOLS_HINT,
            )


def _pii_policy(ctx: SemanticContext) -> Iterator[Diagnostic]:
    if not ctx.spec.policies.pii.fail_closed:
        yield Diagnostic(
            code=Code.E_PII_POLICY,
            message="policies.pii.fail_closed precisa ser true",
            pointer="/policies/pii/fail_closed",
            hint="falha na detecção de PII deve interromper o fluxo, nunca deixar o texto passar",
        )


def _llm_input_trust(ctx: SemanticContext) -> Iterator[Diagnostic]:
    # llm_extract always receives document text, which comes from OCR and is never trusted.
    for graph, node in ctx.nodes():
        if not isinstance(node, LlmExtractNode):
            continue

        agent = ctx.spec.llm_agents.get(node.agent)
        if agent is not None and agent.input_trust != "untrusted":
            yield node_diagnostic(
                graph,
                node,
                Code.E_LLM_EXTRACT_REQUIRES_UNTRUSTED,
                f'o agente "{node.agent}" recebe texto do documento e não pode ser '
                f'input_trust "{agent.input_trust}"',
                field="agent",
                hint=f'defina llm_agents.{node.agent}.input_trust como "untrusted"',
            )
