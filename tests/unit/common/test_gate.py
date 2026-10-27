import pytest

from lab_common.gate import evidence_decision, gate_decision

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((), ("rejected", "not_found")),
        ((1.0,), ("accepted", None)),
        ((0.85,), ("accepted", None)),
        ((0.85, 0.80), ("accepted", None)),
        ((0.80, 0.85), ("accepted", None)),
        ((0.9, 0.86), ("review", "low_margin")),
        ((0.849,), ("review", "below_accept")),
        ((0.6,), ("review", "below_accept")),
        ((0.5999,), ("rejected", "low_confidence")),
    ],
)
def test_rule(scores: tuple[float, ...], expected: tuple[str, str | None]) -> None:
    assert gate_decision(scores, accept=0.85, review=0.6, margin=0.05) == expected


def test_margin_is_rounded_to_the_score_precision() -> None:
    assert 0.85 - 0.80 < 0.05
    assert gate_decision([0.85, 0.80], accept=0.85, review=0.6, margin=0.05)[0] == "accepted"
    assert gate_decision([0.85, 0.8001], accept=0.85, review=0.6, margin=0.05)[0] == "review"


ACCEPTED = ("accepted", None)


@pytest.mark.parametrize(
    ("name", "evidence", "same_best", "expected"),
    [
        (ACCEPTED, ACCEPTED, True, ACCEPTED),
        # "Ferritina" from "Hemograma completo": both clear, different exams.
        (ACCEPTED, ACCEPTED, False, ("review", "evidence_mismatch")),
        # "T4 total" from "T4": the evidence is ambiguous.
        (ACCEPTED, ("review", "low_margin"), True, ("review", "evidence_ambiguous")),
        (ACCEPTED, ("review", "low_margin"), False, ("review", "evidence_ambiguous")),
        (ACCEPTED, ("rejected", "not_found"), False, ("review", "evidence_mismatch")),
        # The evidence never makes an item more certain than its name.
        (("review", "low_margin"), ACCEPTED, True, ("review", "low_margin")),
        (("rejected", "low_confidence"), ACCEPTED, True, ("rejected", "low_confidence")),
    ],
)
def test_model_item_is_accepted_only_when_its_evidence_agrees(
    name: tuple[str, str | None],
    evidence: tuple[str, str | None],
    same_best: bool,
    expected: tuple[str, str | None],
) -> None:
    assert evidence_decision(name, evidence, same_best=same_best) == expected
