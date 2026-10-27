"""`confidence_gate`: the only place where a catalog code is chosen.

For each mention, with `c1` and `c2` the two best candidates from the RAG:

- no candidates -> `rejected` (`not_found`);
- `c1.score >= accept_threshold` and (`c2` absent or margin >= `min_margin`)
  -> `accepted`, `selected = c1`;
- `c1.score >= review_threshold` -> `review` (`low_margin` or `below_accept`);
- otherwise -> `rejected` (`low_confidence`).

The margin is rounded to 4 places, the precision of the scores: in floating
point 0.85 - 0.80 is 0.04999..., which would send a clear case to review.
Accepted items are deduplicated by code (first mention wins). Route: any
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

MARGIN_DIGITS = 4
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


def classify(item: ResolvedExam, params: ConfidenceGateParams) -> GateItem:
    candidates = ranked(item.candidates)
    if not candidates:
        return GateItem(
            mention=item.mention,
            status="rejected",
            selected=None,
            candidates=[],
            reason="not_found",
        )
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin_ok = (
        runner_up is None or round(best.score - runner_up.score, MARGIN_DIGITS) >= params.min_margin
    )
    if best.score >= params.accept_threshold and margin_ok:
        status, selected, reason = "accepted", best, None
    elif best.score >= params.review_threshold:
        reason = "low_margin" if best.score >= params.accept_threshold else "below_accept"
        status, selected = "review", None
    else:
        status, selected, reason = "rejected", None, "low_confidence"
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
