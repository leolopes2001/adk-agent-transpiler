"""`confidence_gate`: the only place where a catalog code is chosen.

Each mention is decided by `lab_common.gate.gate_decision` (the same rule
the RAG evaluation and the calibration script measure); an accepted mention
selects its best candidate. A mention the model named is accepted only if
its evidence resolves, by the same rule, to the same exam
(`lab_common.gate.evidence_decision`); otherwise it goes to review with the
candidates of both searches.
Accepted items are deduplicated by code (first mention wins), and a review
item whose candidates at or above `review_threshold` are all already
accepted is rejected (`already_accepted`): no choice could add an exam, so
asking would only be noise. Route: any
`review` -> `review`; else any `accepted` -> `accepted`; else `empty` with a
`NO_EXAMS_RESOLVED` failure. `selected` is always one of the item's own
candidates, never a value from the model.
"""

from typing import Annotated, Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode
from pydantic import Field, model_validator

from agent_runtime.nodes._common import Params, function_node, parse_input, routed
from agent_runtime.schemas import (
    Candidate,
    GateDecision,
    GateItem,
    ResolvedExam,
    ResolvedExams,
    RunFailure,
)
from lab_common import logging as lab_logging
from lab_common.gate import Decision, evidence_decision, gate_decision

EMPTY_MESSAGE = (
    "Nenhum exame do pedido foi reconhecido no catálogo com confiança suficiente. "
    "Confira o pedido ou procure o atendimento."
)

log = lab_logging.get_logger(__name__)

Threshold = Annotated[float, Field(ge=0, le=1)]


class ConfidenceGateParams(Params):
    accept_threshold: Threshold
    review_threshold: Threshold
    min_margin: Threshold

    @model_validator(mode="after")
    def _ordered(self) -> "ConfidenceGateParams":
        if self.review_threshold > self.accept_threshold:
            raise ValueError("review_threshold must not exceed accept_threshold")
        return self


def ranked(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: (-c.score, c.code))


def _decision(candidates: list[Candidate], params: ConfidenceGateParams) -> Decision:
    return gate_decision(
        (c.score for c in candidates),
        accept=params.accept_threshold,
        review=params.review_threshold,
        margin=params.min_margin,
    )


def _union(first: list[Candidate], second: list[Candidate]) -> list[Candidate]:
    """Both lists ranked together, each code once with its best score."""
    best: dict[str, Candidate] = {}
    for c in [*first, *second]:
        if c.code not in best or c.score > best[c.code].score:
            best[c.code] = c
    return ranked(list(best.values()))


def classify(item: ResolvedExam, params: ConfidenceGateParams) -> GateItem:
    candidates = ranked(item.candidates)
    status, reason = _decision(candidates, params)
    if item.evidence_candidates is not None:
        evidence = ranked(item.evidence_candidates)
        same_best = bool(candidates and evidence) and candidates[0].code == evidence[0].code
        status, reason = evidence_decision(
            (status, reason), _decision(evidence, params), same_best=same_best
        )
        if status == "review" and reason in {"evidence_ambiguous", "evidence_mismatch"}:
            candidates = _union(candidates, evidence)
    selected = candidates[0] if status == "accepted" else None
    gate_item = GateItem.model_validate(
        {
            "mention": item.mention,
            "status": status,
            "selected": selected,
            "candidates": candidates,
            "reason": reason,
        }
    )
    # Not an assert: it must hold with -O too.
    if gate_item.selected is not None and gate_item.selected not in gate_item.candidates:
        raise RuntimeError("gate invariant broken: selected is not a candidate")
    return gate_item


def _unless_covered(
    item: GateItem, accepted_codes: set[str], params: ConfidenceGateParams
) -> GateItem:
    if item.status != "review":
        return item
    options = {c.code for c in item.candidates if c.score >= params.review_threshold}
    if options and options <= accepted_codes:
        return item.model_copy(update={"status": "rejected", "reason": "already_accepted"})
    return item


def decide(resolved: ResolvedExams, params: ConfidenceGateParams) -> tuple[GateDecision, str]:
    """Decision for every mention and the route it takes."""
    items: list[GateItem] = []
    accepted_codes: set[str] = set()
    for resolved_item in resolved.items:
        item = classify(resolved_item, params)
        if item.selected is not None:
            if item.selected.code in accepted_codes:
                continue
            accepted_codes.add(item.selected.code)
        items.append(item)
    items = [_unless_covered(item, accepted_codes, params) for item in items]
    statuses = {item.status for item in items}
    route = "review" if "review" in statuses else "accepted" if "accepted" in statuses else "empty"
    return GateDecision(items=items), route


def confidence_gate(
    *, name: str, accept_threshold: float, review_threshold: float, min_margin: float
) -> FunctionNode:
    """Classifies candidates; routes `accepted`, `review` or `empty`."""
    params = ConfidenceGateParams(
        accept_threshold=accept_threshold,
        review_threshold=review_threshold,
        min_margin=min_margin,
    )

    async def run(ctx: Context, node_input: Any) -> Event:
        decision, route = decide(parse_input(ResolvedExams, node_input), params)
        log.info(
            "gate_decision",
            node=name,
            route=route,
            accepted=[item.selected.code for item in decision.items if item.selected],
            review=sum(item.status == "review" for item in decision.items),
            rejected=sum(item.status == "rejected" for item in decision.items),
        )
        if route == "empty":
            return routed(RunFailure(code="NO_EXAMS_RESOLVED", user_message=EMPTY_MESSAGE), route)
        return routed(decision, route)

    return function_node(name, run)
