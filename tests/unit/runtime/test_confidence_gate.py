import pytest
from hypothesis import given
from hypothesis import strategies as st
from runtime_fakes import candidate, run_node

from agent_runtime.nodes.confidence_gate import (
    ConfidenceGateParams,
    classify,
    confidence_gate,
    decide,
)
from agent_runtime.schemas import (
    Candidate,
    ExamMention,
    GateDecision,
    ResolvedExam,
    ResolvedExams,
    RunFailure,
)

pytestmark = pytest.mark.unit

PARAMS = ConfidenceGateParams(accept_threshold=0.85, review_threshold=0.6, min_margin=0.05)


def cand(code: str, score: float) -> Candidate:
    return Candidate.model_validate(candidate(code, f"Exame {code}", score))


def resolved(text: str, *candidates: Candidate) -> ResolvedExam:
    mention = ExamMention(text=text, source="both", evidence=text)
    return ResolvedExam(mention=mention, candidates=list(candidates))


@pytest.mark.parametrize(
    ("scores", "status", "selected", "reason"),
    [
        ([], "rejected", None, "not_found"),
        ([1.0], "accepted", "LAB-10001", None),
        ([0.85], "accepted", "LAB-10001", None),
        ([0.85, 0.80], "accepted", "LAB-10001", None),
        ([0.9, 0.86], "review", None, "low_margin"),
        ([0.849], "review", None, "below_accept"),
        ([0.6], "review", None, "below_accept"),
        ([0.5999], "rejected", None, "low_confidence"),
        ([0.2, 0.1], "rejected", None, "low_confidence"),
    ],
)
def test_gate_table(
    scores: list[float], status: str, selected: str | None, reason: str | None
) -> None:
    candidates = [cand(f"LAB-1000{i + 1}", s) for i, s in enumerate(scores)]

    item = classify(resolved("x", *candidates), PARAMS)

    assert (item.status, item.reason) == (status, reason)
    assert (item.selected.code if item.selected else None) == selected


def test_margin_is_rounded_like_the_scores() -> None:
    # 0.85 - 0.80 == 0.04999... in floating point.
    assert 0.85 - 0.80 < 0.05
    item = classify(resolved("x", cand("LAB-10001", 0.85), cand("LAB-10002", 0.80)), PARAMS)

    assert item.status == "accepted"


def test_candidates_are_ranked_whatever_order_they_come_in() -> None:
    item = classify(resolved("x", cand("LAB-10002", 0.7), cand("LAB-10009", 1.0)), PARAMS)

    assert item.selected == cand("LAB-10009", 1.0)
    assert [c.code for c in item.candidates] == ["LAB-10009", "LAB-10002"]


def test_accepted_items_are_deduplicated_by_code() -> None:
    decision, route = decide(
        ResolvedExams(
            items=[
                resolved("PCR", cand("LAB-10037", 1.0)),
                resolved("Proteína C reativa", cand("LAB-10037", 1.0)),
                resolved("TSH", cand("LAB-10043", 1.0)),
            ]
        ),
        PARAMS,
    )

    assert [item.mention.text for item in decision.items] == ["PCR", "TSH"]
    assert route == "accepted"


def test_review_item_covered_by_accepted_codes_is_not_asked() -> None:
    decision, route = decide(
        ResolvedExams(
            items=[
                resolved("TSH", cand("LAB-10043", 1.0)),
                resolved("T4 livre", cand("LAB-10044", 1.0)),
                resolved(
                    "TSH + T4L",
                    cand("LAB-10043", 0.8),
                    cand("LAB-10044", 0.8),
                    cand("LAB-10001", 0.3),
                ),
                resolved("T4", cand("LAB-10044", 0.8), cand("LAB-10045", 0.8)),
            ]
        ),
        PARAMS,
    )

    by_text = {item.mention.text: (item.status, item.reason) for item in decision.items}
    assert by_text["TSH + T4L"] == ("rejected", "already_accepted")
    assert by_text["T4"] == ("review", "below_accept")
    assert route == "review"


@pytest.mark.parametrize(
    ("scores", "route"),
    [
        ([[1.0], [0.7]], "review"),
        ([[1.0], [0.1]], "accepted"),
        ([[0.1], []], "empty"),
        ([], "empty"),
    ],
)
def test_route(scores: list[list[float]], route: str) -> None:
    items = [
        resolved(f"m{i}", *[cand(f"LAB-2000{i}", s) for s in found])
        for i, found in enumerate(scores)
    ]

    assert decide(ResolvedExams(items=items), PARAMS)[1] == route


def test_params_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="review_threshold"):
        ConfidenceGateParams(accept_threshold=0.6, review_threshold=0.85, min_margin=0.05)


async def test_node_routes_and_outputs() -> None:
    node = confidence_gate(
        name="gate", accept_threshold=0.85, review_threshold=0.6, min_margin=0.05
    )

    accepted = await run_node(node, ResolvedExams(items=[resolved("TSH", cand("LAB-10043", 1.0))]))
    empty = await run_node(node, ResolvedExams(items=[resolved("x")]))

    assert accepted.route == "accepted"
    assert GateDecision.model_validate(accepted.output).items[0].status == "accepted"
    assert empty.route == "empty"
    assert RunFailure.model_validate(empty.output).code == "NO_EXAMS_RESOLVED"


SCORES = st.floats(min_value=0, max_value=1, allow_nan=False).map(lambda s: round(s, 4))
CANDIDATES = st.lists(
    st.tuples(st.integers(min_value=10000, max_value=10200), SCORES), max_size=6
).map(lambda pairs: [cand(f"LAB-{code}", score) for code, score in pairs])
THRESHOLDS = st.tuples(SCORES, SCORES, SCORES).map(
    lambda t: ConfidenceGateParams(
        accept_threshold=max(t[0], t[1]), review_threshold=min(t[0], t[1]), min_margin=t[2]
    )
)


@given(st.lists(CANDIDATES, max_size=6), THRESHOLDS)
def test_selected_is_always_one_of_the_rag_candidates(
    all_candidates: list[list[Candidate]], params: ConfidenceGateParams
) -> None:
    items = [resolved(f"m{i}", *found) for i, found in enumerate(all_candidates)]

    decision, route = decide(ResolvedExams(items=items), params)

    offered = {c.model_dump_json() for found in all_candidates for c in found}
    codes = [item.selected.code for item in decision.items if item.selected]
    assert len(codes) == len(set(codes))
    for item in decision.items:
        assert (item.selected is not None) == (item.status == "accepted")
        if item.selected is not None:
            assert item.selected in item.candidates
            assert item.selected.model_dump_json() in offered
            assert item.selected.score >= params.accept_threshold
    statuses = {item.status for item in decision.items}
    assert route == (
        "review" if "review" in statuses else "accepted" if "accepted" in statuses else "empty"
    )
