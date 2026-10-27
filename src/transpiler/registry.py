"""Node kind registry: the single source of truth for validation and lowering.

Each kind declares the port types it accepts and produces (per route for
routers), its params model, which references it needs, which ADK primitive it
becomes and which runtime factory builds it. Port types are names only; the
classes live in `agent_runtime.schemas` and a contract test keeps both sides in
sync.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel

from transpiler.spec import models


class PortType(StrEnum):
    RUN_INPUT = "RunInput"
    OCR_DOCUMENT = "OcrDocument"
    GUARDED_DOCUMENT = "GuardedDocument"
    LLM_MENTIONS = "LlmMentions"
    EXAM_MENTIONS = "ExamMentions"
    JOIN_RESULT = "JoinResult"
    RESOLVED_EXAMS = "ResolvedExams"
    GATE_DECISION = "GateDecision"
    SCHEDULE_OUTCOME = "ScheduleOutcome"
    RUN_FAILURE = "RunFailure"


class NodeKind(StrEnum):
    OCR_EXTRACT = "ocr_extract"
    UNTRUSTED_TEXT_GUARD = "untrusted_text_guard"
    LLM_EXTRACT = "llm_extract"
    LEXICAL_EXTRACT = "lexical_extract"
    JOIN = "join"
    RECONCILE = "reconcile"
    RAG_RESOLVE = "rag_resolve"
    CONFIDENCE_GATE = "confidence_gate"
    HUMAN_REVIEW = "human_review"
    SCHEDULE = "schedule"
    PRESENT = "present"
    ABORT = "abort"
    SUBWORKFLOW = "subworkflow"


type Requirement = Literal["server", "tool", "service", "agent", "ref"]
type AdkPrimitive = Literal["function", "agent", "join", "workflow"]


@dataclass(frozen=True, slots=True)
class NodeKindSpec:
    kind: NodeKind
    input_types: frozenset[PortType]
    """Accepted input types; empty means the kind accepts anything (join, subworkflow)."""
    adk_primitive: AdkPrimitive
    factory: str | None
    """Name of the `agent_runtime.nodes` factory; for llm_extract, the factory of its input node."""
    output_type: PortType | None = None
    routes: Mapping[str, PortType] = field(default_factory=lambda: MappingProxyType({}))
    params_model: type[BaseModel] | None = None
    requires: frozenset[Requirement] = frozenset()
    terminal: bool = False
    accepts_retry: bool = False
    """Calls a server, service or model, so it takes retry and timeout_s."""
    rerun_on_resume: bool = False
    join_inputs: tuple[PortType, ...] = ()
    """For kinds fed by a join: the exact types the join must combine, in sorted order."""
    tools: frozenset[str] = frozenset()
    """MCP tools the runtime factory knows how to call, for kinds that require a tool."""

    @property
    def is_router(self) -> bool:
        return bool(self.routes)

    def output_types(self) -> frozenset[PortType]:
        """Every type this kind can emit, across routes. Empty for terminals and subworkflows."""
        if self.routes:
            return frozenset(self.routes.values())
        return frozenset() if self.output_type is None else frozenset({self.output_type})


def _routes(**routes: PortType) -> Mapping[str, PortType]:
    return MappingProxyType(routes)


_T = PortType

REGISTRY: Mapping[NodeKind, NodeKindSpec] = MappingProxyType(
    {
        spec.kind: spec
        for spec in (
            NodeKindSpec(
                kind=NodeKind.OCR_EXTRACT,
                input_types=frozenset({_T.RUN_INPUT}),
                routes=_routes(ok=_T.OCR_DOCUMENT, low_quality=_T.RUN_FAILURE),
                params_model=models.OcrExtractParams,
                requires=frozenset({"server", "tool"}),
                adk_primitive="function",
                factory="ocr_extract",
                accepts_retry=True,
                tools=frozenset({"extract_text"}),
            ),
            NodeKindSpec(
                kind=NodeKind.UNTRUSTED_TEXT_GUARD,
                input_types=frozenset({_T.OCR_DOCUMENT}),
                routes=_routes(clean=_T.GUARDED_DOCUMENT, blocked=_T.RUN_FAILURE),
                params_model=models.UntrustedTextGuardParams,
                adk_primitive="function",
                factory="untrusted_text_guard",
            ),
            NodeKindSpec(
                kind=NodeKind.LLM_EXTRACT,
                input_types=frozenset({_T.GUARDED_DOCUMENT}),
                output_type=_T.LLM_MENTIONS,
                requires=frozenset({"agent"}),
                adk_primitive="agent",
                factory="llm_input",
                accepts_retry=True,
            ),
            NodeKindSpec(
                kind=NodeKind.LEXICAL_EXTRACT,
                input_types=frozenset({_T.GUARDED_DOCUMENT}),
                output_type=_T.EXAM_MENTIONS,
                params_model=models.LexicalExtractParams,
                requires=frozenset({"server", "tool"}),
                adk_primitive="function",
                factory="lexical_extract",
                accepts_retry=True,
                tools=frozenset({"batch_search"}),
            ),
            NodeKindSpec(
                kind=NodeKind.JOIN,
                input_types=frozenset(),
                output_type=_T.JOIN_RESULT,
                adk_primitive="join",
                factory=None,
            ),
            NodeKindSpec(
                kind=NodeKind.RECONCILE,
                input_types=frozenset({_T.JOIN_RESULT}),
                output_type=_T.EXAM_MENTIONS,
                params_model=models.ReconcileParams,
                adk_primitive="function",
                factory="reconcile",
                join_inputs=tuple(sorted((_T.LLM_MENTIONS, _T.EXAM_MENTIONS))),
            ),
            NodeKindSpec(
                kind=NodeKind.RAG_RESOLVE,
                input_types=frozenset({_T.EXAM_MENTIONS}),
                output_type=_T.RESOLVED_EXAMS,
                params_model=models.RagResolveParams,
                requires=frozenset({"server", "tool"}),
                adk_primitive="function",
                factory="rag_resolve",
                accepts_retry=True,
                tools=frozenset({"batch_search"}),
            ),
            NodeKindSpec(
                kind=NodeKind.CONFIDENCE_GATE,
                input_types=frozenset({_T.RESOLVED_EXAMS}),
                routes=_routes(
                    accepted=_T.GATE_DECISION, review=_T.GATE_DECISION, empty=_T.RUN_FAILURE
                ),
                params_model=models.ConfidenceGateParams,
                adk_primitive="function",
                factory="confidence_gate",
            ),
            NodeKindSpec(
                kind=NodeKind.HUMAN_REVIEW,
                input_types=frozenset({_T.GATE_DECISION}),
                routes=_routes(confirmed=_T.GATE_DECISION, rejected=_T.GATE_DECISION),
                params_model=models.HumanReviewParams,
                adk_primitive="function",
                factory="human_review",
                rerun_on_resume=True,
            ),
            NodeKindSpec(
                kind=NodeKind.SCHEDULE,
                input_types=frozenset({_T.GATE_DECISION}),
                output_type=_T.SCHEDULE_OUTCOME,
                requires=frozenset({"service"}),
                adk_primitive="function",
                factory="schedule",
                accepts_retry=True,
            ),
            NodeKindSpec(
                kind=NodeKind.PRESENT,
                input_types=frozenset({_T.GATE_DECISION, _T.SCHEDULE_OUTCOME}),
                params_model=models.PresentParams,
                adk_primitive="function",
                factory="present",
                terminal=True,
            ),
            NodeKindSpec(
                kind=NodeKind.ABORT,
                input_types=frozenset({_T.RUN_FAILURE}),
                adk_primitive="function",
                factory="abort",
                terminal=True,
            ),
            NodeKindSpec(
                kind=NodeKind.SUBWORKFLOW,
                input_types=frozenset(),
                requires=frozenset({"ref"}),
                adk_primitive="workflow",
                factory=None,
            ),
        )
    }
)


def kind_spec(kind: str) -> NodeKindSpec:
    return REGISTRY[NodeKind(kind)]
