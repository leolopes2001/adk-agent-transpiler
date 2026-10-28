import uuid
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from google.adk import Context, Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode
from pydantic import Field

from agent_runtime.nodes._common import Params, function_node, parse_input, routed
from agent_runtime.schemas import Candidate, GateDecision, GateItem
from lab_common import logging as lab_logging

NONE_CHOICE = "none"
REVIEW_MESSAGE = "Escolha o exame correto para cada item, ou 'none' se nenhum corresponder."
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"choices": {"type": "object", "additionalProperties": {"type": "string"}}},
    "required": ["choices"],
}
_DEFAULT_MAX_CANDIDATES = 3

log = lab_logging.get_logger(__name__)


class HumanReviewParams(Params):
    on_non_interactive: Literal["reject"] = "reject"
    max_candidates: Annotated[int, Field(ge=1, le=10)] = _DEFAULT_MAX_CANDIDATES


def human_review(
    *,
    name: str,
    on_non_interactive: Literal["reject"] = "reject",
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> FunctionNode:
    params = HumanReviewParams(on_non_interactive=on_non_interactive, max_candidates=max_candidates)

    async def run(ctx: Context, node_input: Any) -> Event | RequestInput:
        decision = parse_input(GateDecision, node_input)
        pending = sum(item.status == "review" for item in decision.items)
        interactive = ctx.state.get("run.interactive") is True
        if pending and interactive:
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
            interactive=interactive,
            reasons=sorted(item.reason for item in decision.items if item.reason),
        )
        return routed(decision, _route_after_review(decision))

    return function_node(name, run, rerun_on_resume=True)


def interrupt_id(ctx: Context) -> str:
    run_id = ctx.state.get("run.id")
    try:
        return f"review-{uuid.UUID(str(run_id))}"
    except ValueError:
        return f"review-{ctx.session.id}"


def review_payload(decision: GateDecision, max_candidates: int) -> dict[str, Any]:
    return {
        "items": [
            {
                "index": index,
                "mention": item.mention.text,
                "evidence": item.mention.evidence,
                "source": item.mention.source,
                "options": [
                    {"code": c.code, "name": c.name, "score": c.score}
                    for c in item.candidates[:max_candidates]
                ],
            }
            for index, item in enumerate(decision.items)
            if item.status == "review"
        ]
    }


def apply_choices(decision: GateDecision, response: object, max_candidates: int) -> GateDecision:
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if not isinstance(choices, Mapping):
        choices = {}

    accepted_codes = {item.selected.code for item in decision.items if item.selected}
    items: list[GateItem] = []
    for index, item in enumerate(decision.items):
        if item.status != "review":
            items.append(item)
            continue

        offered = {c.code: c for c in item.candidates[:max_candidates]}
        settled = _settle_item(item, choices.get(str(index)), offered, accepted_codes)
        if settled.selected is not None:
            accepted_codes.add(settled.selected.code)
        items.append(settled)

    return GateDecision(items=items)


def reject_all(decision: GateDecision) -> GateDecision:
    return GateDecision(
        items=[
            _rejected(item, "needs_human_review") if item.status == "review" else item
            for item in decision.items
        ]
    )


# The operator picks one of the offered candidates and never types a code.
def _settle_item(
    item: GateItem, choice: object, offered: Mapping[str, Candidate], accepted_codes: set[str]
) -> GateItem:
    if choice == NONE_CHOICE:
        return _rejected(item, "rejected_by_reviewer")

    if not isinstance(choice, str) or choice not in offered:
        return _rejected(item, "invalid_review_response")

    # Rejected, not dropped: every mention of the order stays in the result.
    if choice in accepted_codes:
        return _rejected(item, "already_accepted")

    return item.model_copy(
        update={"status": "accepted", "selected": offered[choice], "reason": "reviewed"}
    )


def _rejected(item: GateItem, reason: str) -> GateItem:
    return item.model_copy(update={"status": "rejected", "selected": None, "reason": reason})


def _route_after_review(decision: GateDecision) -> str:
    return "confirmed" if any(item.status == "accepted" for item in decision.items) else "rejected"
