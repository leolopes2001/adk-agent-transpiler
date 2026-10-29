from typing import Any

import pytest
from runtime_fakes import candidate, run_node, start_node

from agent_runtime.nodes.human_review import (
    apply_choices,
    human_review,
    reject_all,
    review_payload,
)
from agent_runtime.schemas import Candidate, ExamMention, GateDecision, GateItem

pytestmark = pytest.mark.unit

RUN_ID = "2f3b5fd6-2866-4383-9a4e-6d1f0b8c7a11"
T4_FREE = Candidate.model_validate(candidate("LAB-10044", "T4 livre", 0.8))
T4_TOTAL = Candidate.model_validate(candidate("LAB-10045", "T4 total", 0.8))
T3 = Candidate.model_validate(candidate("LAB-10046", "T3", 0.7))
OTHER = Candidate.model_validate(candidate("LAB-10047", "T3 livre", 0.6))
HEMOGRAMA = Candidate.model_validate(candidate("LAB-10001", "Hemograma completo", 1.0))


def _item(text: str, status: str, *candidates: Candidate) -> GateItem:
    selected = candidates[0] if status == "accepted" else None
    reason = None if status == "accepted" else "low_margin"
    return GateItem.model_validate(
        {
            "mention": ExamMention(text=text, source="both", evidence=text),
            "status": status,
            "selected": selected,
            "candidates": list(candidates),
            "reason": reason,
        }
    )


DECISION = GateDecision(
    items=[
        _item("Hemograma completo", "accepted", HEMOGRAMA),
        _item("T4", "review", T4_FREE, T4_TOTAL, T3, OTHER),
    ]
)


def _statuses(decision: GateDecision) -> list[tuple[str, str | None, str | None]]:
    return [
        (item.status, item.selected.code if item.selected else None, item.reason)
        for item in decision.items
    ]


def test_payload_offers_at_most_max_candidates() -> None:
    payload = review_payload(DECISION, max_candidates=3)

    [item] = payload["items"]
    assert item["index"] == 1
    assert item["mention"] == "T4"
    assert [o["code"] for o in item["options"]] == ["LAB-10044", "LAB-10045", "LAB-10046"]
    assert item["source"] == "both"
    assert set(item["options"][0]) == {"code", "name", "score"}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"choices": {"1": "LAB-10045"}}, ("accepted", "LAB-10045", "reviewed")),
        ({"choices": {"1": "none"}}, ("rejected", None, "rejected_by_reviewer")),
        ({"choices": {"1": "LAB-10047"}}, ("rejected", None, "invalid_review_response")),
        ({"choices": {"1": "LAB-99999"}}, ("rejected", None, "invalid_review_response")),
        ({"choices": {"1": 3}}, ("rejected", None, "invalid_review_response")),
        ({"choices": {}}, ("rejected", None, "invalid_review_response")),
        ({"choices": "LAB-10044"}, ("rejected", None, "invalid_review_response")),
        ("LAB-10044", ("rejected", None, "invalid_review_response")),
    ],
)
def test_apply_choices(response: Any, expected: tuple[str, str | None, str | None]) -> None:
    decision = apply_choices(DECISION, response, max_candidates=3)

    assert _statuses(decision) == [("accepted", "LAB-10001", None), expected]
    for item in decision.items:
        assert item.selected is None or item.selected in item.candidates


def test_choice_of_an_already_accepted_code_keeps_the_item_as_already_accepted() -> None:
    decision = GateDecision(
        items=[
            _item("Hemograma completo", "accepted", HEMOGRAMA),
            _item("Hemograma", "review", HEMOGRAMA, T4_FREE),
        ]
    )

    reviewed = apply_choices(decision, {"choices": {"1": "LAB-10001"}}, 3)

    assert _statuses(reviewed) == [
        ("accepted", "LAB-10001", None),
        ("rejected", None, "already_accepted"),
    ]
    assert [item.mention.text for item in reviewed.items] == ["Hemograma completo", "Hemograma"]


def test_two_review_items_choosing_the_same_code_keep_both_lines() -> None:
    decision = GateDecision(
        items=[
            _item("T4", "review", T4_FREE, T4_TOTAL),
            _item("T4 L", "review", T4_FREE, T4_TOTAL),
        ]
    )

    reviewed = apply_choices(decision, {"choices": {"0": "LAB-10044", "1": "LAB-10044"}}, 3)

    assert _statuses(reviewed) == [
        ("accepted", "LAB-10044", "reviewed"),
        ("rejected", None, "already_accepted"),
    ]


def test_reject_all() -> None:
    assert _statuses(reject_all(DECISION)) == [
        ("accepted", "LAB-10001", None),
        ("rejected", None, "needs_human_review"),
    ]


async def test_non_interactive_rejects_review_items_and_keeps_accepted() -> None:
    run = await run_node(human_review(name="review"), DECISION, {"run.interactive": False})

    assert run.route == "confirmed"
    assert _statuses(GateDecision.model_validate(run.output))[1][2] == "needs_human_review"


async def test_non_interactive_with_nothing_accepted_is_rejected() -> None:
    only_review = GateDecision(items=[DECISION.items[1]])

    run = await run_node(human_review(name="review"), only_review)

    assert run.route == "rejected"


async def test_interactive_asks_once_and_applies_the_answer_on_resume() -> None:
    state = {"run.interactive": True, "run.id": RUN_ID}
    session = await start_node(human_review(name="review", max_candidates=2), DECISION, state)

    interrupt = session.interrupt()
    assert interrupt is not None
    assert interrupt.long_running_tool_ids == {f"review-{RUN_ID}"}
    assert interrupt.content is not None
    assert interrupt.content.parts
    call = interrupt.content.parts[0].function_call
    assert call is not None
    assert call.args is not None
    assert [o["code"] for o in call.args["payload"]["items"][0]["options"]] == [
        "LAB-10044",
        "LAB-10045",
    ]
    assert (await session.result()).route is None

    resumed = await session.resume({"choices": {"1": "LAB-10044"}})

    assert resumed.route == "confirmed"
    assert _statuses(GateDecision.model_validate(resumed.output))[1] == (
        "accepted",
        "LAB-10044",
        "reviewed",
    )


async def test_interactive_none_for_everything_is_rejected() -> None:
    only_review = GateDecision(items=[DECISION.items[1]])
    session = await start_node(
        human_review(name="review"), only_review, {"run.interactive": True, "run.id": RUN_ID}
    )

    resumed = await session.resume({"choices": {"0": "none"}})

    assert resumed.route == "rejected"


async def test_decision_without_review_items_passes_through() -> None:
    only_accepted = GateDecision(items=[DECISION.items[0]])

    run = await run_node(human_review(name="review"), only_accepted, {"run.interactive": True})

    assert run.route == "confirmed"
