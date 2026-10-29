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
        (ACCEPTED, (1.0,), True, ACCEPTED),
        # "Trigliceríideos": an OCR typo keeps the evidence below `accept`.
        (ACCEPTED, (0.7241, 0.5063), True, ACCEPTED),
        (ACCEPTED, (0.6,), True, ACCEPTED),
        # "Ferritina" from "Hemograma completo": both clear, different exams.
        (ACCEPTED, (1.0,), False, ("review", "evidence_mismatch")),
        (ACCEPTED, (0.7241, 0.5063), False, ("review", "evidence_mismatch")),
        # "T4 total" from "T4": the evidence is ambiguous, above or below `accept`.
        (ACCEPTED, (0.925, 0.925), True, ("review", "evidence_ambiguous")),
        (ACCEPTED, (0.925, 0.925), False, ("review", "evidence_ambiguous")),
        (ACCEPTED, (0.7, 0.68), True, ("review", "evidence_ambiguous")),
        (ACCEPTED, (0.5999,), True, ("review", "evidence_mismatch")),
        (ACCEPTED, (), False, ("review", "evidence_mismatch")),
        # The evidence never makes an item more certain than its name.
        (("review", "low_margin"), (1.0,), True, ("review", "low_margin")),
        (("rejected", "low_confidence"), (1.0,), True, ("rejected", "low_confidence")),
    ],
)
def test_model_item_is_accepted_only_when_its_evidence_agrees(
    name: tuple[str, str | None],
    evidence: tuple[float, ...],
    same_best: bool,
    expected: tuple[str, str | None],
) -> None:
    decision = evidence_decision(name, evidence, same_best=same_best, review=0.6, margin=0.05)

    assert decision == expected
