"""`human_review`: an operator settles the items the gate sent to review.

Interactive runs (`run.interactive`) interrupt once with a `RequestInput`
(`interrupt_id = "review-<run id>"`) listing, per item, up to
`max_candidates` options taken from the item's own candidates. The node
reruns on resume and reads the answer `{"choices": {"<index>": "<code>" |
"none"}}`. A choice outside the offered codes rejects the item
(`invalid_review_response`): the operator picks, never types a code. A
choice of a code another item already accepted keeps the item, rejected as
`already_accepted`: every mention of the order stays in the result.

Non-interactive runs reject review items (`needs_human_review`). Route:
`confirmed` when at least one item ends accepted, else `rejected`.
"""

import uuid
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from google.adk import Context, Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode
from pydantic import Field

from agent_runtime.nodes._common import Params, function_node, parse_input, routed
from agent_runtime.schemas import GateDecision, GateItem
from lab_common import logging as lab_logging

NONE_CHOICE = "none"
REVIEW_MESSAGE = "Escolha o exame correto para cada item, ou 'none' se nenhum corresponder."
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"choices": {"type": "object", "additionalProperties": {"type": "string"}}},
    "required": ["choices"],
}

log = lab_logging.get_logger(__name__)


class HumanReviewParams(Params):
    on_non_interactive: Literal["reject"] = "reject"
    max_candidates: Annotated[int, Field(ge=1, le=10)] = 3


def interrupt_id(ctx: Context) -> str:
    run_id = ctx.state.get("run.id")
    try:
        return f"review-{uuid.UUID(str(run_id))}"
    except ValueError:
        return f"review-{ctx.session.id}"


def review_payload(decision: GateDecision, max_candidates: int) -> dict[str, Any]:
    """What the operator sees: mention text (masked) and the candidate options."""
    return {
        "items": [
            {
                "index": index,
                "mention": item.mention.text,
                "evidence": item.mention.evidence,
                "options": [
                    {"code": c.code, "name": c.name, "score": c.score}
                    for c in item.candidates[:max_candidates]
                ],
            }
            for index, item in enumerate(decision.items)
            if item.status == "review"
        ]
    }


def _rejected(item: GateItem, reason: str) -> GateItem:
    return item.model_copy(update={"status": "rejected", "selected": None, "reason": reason})


def apply_choices(decision: GateDecision, response: object, max_candidates: int) -> GateDecision:
    """Decision after the operator's answer; anything unexpected rejects the item."""
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if not isinstance(choices, Mapping):
        choices = {}
    accepted_codes = {item.selected.code for item in decision.items if item.selected}
    items: list[GateItem] = []
    for index, item in enumerate(decision.items):
        if item.status != "review":
            items.append(item)
            continue
        choice = choices.get(str(index))
        offered = {c.code: c for c in item.candidates[:max_candidates]}
        if choice == NONE_CHOICE:
            items.append(_rejected(item, "rejected_by_reviewer"))
        elif not isinstance(choice, str) or choice not in offered:
            items.append(_rejected(item, "invalid_review_response"))
        elif choice in accepted_codes:
            items.append(_rejected(item, "already_accepted"))
        else:
            accepted_codes.add(choice)
            items.append(
                item.model_copy(
                    update={"status": "accepted", "selected": offered[choice], "reason": "reviewed"}
                )
            )
    return GateDecision(items=items)


def reject_all(decision: GateDecision) -> GateDecision:
    return GateDecision(
        items=[
            _rejected(item, "needs_human_review") if item.status == "review" else item
            for item in decision.items
        ]
    )


def _route(decision: GateDecision) -> str:
    return "confirmed" if any(item.status == "accepted" for item in decision.items) else "rejected"


def human_review(
    *, name: str, on_non_interactive: Literal["reject"] = "reject", max_candidates: int = 3
) -> FunctionNode:
    """Asks the operator to choose among candidates; routes `confirmed` or `rejected`."""
    params = HumanReviewParams(on_non_interactive=on_non_interactive, max_candidates=max_candidates)

    async def run(ctx: Context, node_input: Any) -> Event | RequestInput:
        decision = parse_input(GateDecision, node_input)
        pending = sum(item.status == "review" for item in decision.items)
        if pending and ctx.state.get("run.interactive") is True:
            key = interrupt_id(ctx)
            if key not in ctx.resume_inputs:
                log.info("review_requested", node=name, items=pending)
                return RequestInput(
                    interrupt_id=key,
                    message=REVIEW_MESSAGE,
                    payload=review_payload(decision, params.max_candidates),
                    response_schema=RESPONSE_SCHEMA,
                )
            decision = apply_choices(decision, ctx.resume_inputs[key], params.max_candidates)
        elif pending:
            decision = reject_all(decision)
        log.info(
            "review_resolved",
            node=name,
            interactive=ctx.state.get("run.interactive") is True,
            reasons=sorted(item.reason for item in decision.items if item.reason),
        )
        return routed(decision, _route(decision))

    return function_node(name, run, rerun_on_resume=True)
