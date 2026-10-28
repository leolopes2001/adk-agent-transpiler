from collections.abc import Iterable
from typing import Literal

# RAG scores have 4 places; unrounded, 0.85 - 0.80 is 0.0499... and fails a 0.05 margin.
MARGIN_DIGITS = 4

GateStatus = Literal["accepted", "review", "rejected"]
GateReason = Literal[
    "not_found",
    "low_margin",
    "below_accept",
    "low_confidence",
    "evidence_ambiguous",
    "evidence_mismatch",
]
type Decision = tuple[GateStatus, GateReason | None]


def gate_decision(
    scores: Iterable[float], *, accept: float, review: float, margin: float
) -> Decision:
    ranked = sorted(scores, reverse=True)
    if not ranked:
        return "rejected", "not_found"

    best = ranked[0]
    if best >= accept and _has_clear_margin(ranked, margin):
        return "accepted", None

    if best >= review:
        return "review", "low_margin" if best >= accept else "below_accept"

    return "rejected", "low_confidence"


def evidence_decision(
    name: Decision,
    evidence_scores: Iterable[float],
    *,
    same_best: bool,
    review: float,
    margin: float,
) -> Decision:
    if name[0] != "accepted":
        return name

    # Evidence only needs `review`: an OCR typo must not send an exam both searches agree on.
    ranked = sorted(evidence_scores, reverse=True)
    if not ranked or ranked[0] < review:
        return "review", "evidence_mismatch"

    if not _has_clear_margin(ranked, margin):
        return "review", "evidence_ambiguous"

    return name if same_best else ("review", "evidence_mismatch")


def _has_clear_margin(ranked: list[float], margin: float) -> bool:
    best, *rest = ranked
    return not rest or round(best - rest[0], MARGIN_DIGITS) >= margin
