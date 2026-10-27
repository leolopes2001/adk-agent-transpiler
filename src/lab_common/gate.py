"""The confidence gate rule, as a pure function of candidate scores.

The runtime's `confidence_gate` node, the RAG regression test and the
threshold calibration script all decide through `gate_decision`, so the rule
they measure is the rule that runs. With `best` and `runner_up` the two
highest scores:

- no candidates -> `rejected` (`not_found`);
- `best >= accept` and (no runner-up or `round(best - runner_up, 4) >= margin`)
  -> `accepted`;
- `best >= review` -> `review` (`low_margin` when it reached `accept`,
  otherwise `below_accept`);
- otherwise -> `rejected` (`low_confidence`).

The margin is rounded to 4 places, the precision of the RAG scores: in
floating point 0.85 - 0.80 is 0.04999..., which would send a clear case to
review.

A mention named by the model is also resolved by its evidence (the text it
copied from the document), and `evidence_decision` combines both: an
accepted name stays accepted when the evidence ranks the same exam first,
with `best >= review` and the margin kept. The evidence need not reach
`accept`, so an OCR typo ("Trigliceríideos") does not send an exam both
searches agree on to review. Otherwise the name goes to `review`
(`evidence_ambiguous` when the evidence's top two are within the margin, as
"T4" for "T4 total"; `evidence_mismatch` when it points elsewhere or is
below `review`, as "Hemograma completo" for "Ferritina"). The evidence never
makes a mention more certain than its name.
"""

from collections.abc import Iterable
from typing import Literal

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
    """Status and reason for a mention whose candidates have `scores` (any order)."""
    ranked = sorted(scores, reverse=True)
    if not ranked:
        return "rejected", "not_found"
    best = ranked[0]
    if best >= accept and _margin_ok(ranked, margin):
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
    """Decision for a model mention from its name's decision and its evidence's scores.

    `same_best` tells whether both searches ranked the same exam first.
    """
    if name[0] != "accepted":
        return name
    ranked = sorted(evidence_scores, reverse=True)
    if not ranked or ranked[0] < review:
        return "review", "evidence_mismatch"
    if not _margin_ok(ranked, margin):
        return "review", "evidence_ambiguous"
    return name if same_best else ("review", "evidence_mismatch")


def _margin_ok(ranked: list[float], margin: float) -> bool:
    best, *rest = ranked
    return not rest or round(best - rest[0], MARGIN_DIGITS) >= margin
