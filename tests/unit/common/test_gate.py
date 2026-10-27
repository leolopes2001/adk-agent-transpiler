import pytest

from lab_common.gate import gate_decision

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
