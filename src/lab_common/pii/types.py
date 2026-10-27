"""Core types of the PII engine.

A detector reports where personal data is, never what it is: `PiiSpan` holds
offsets into the text it was given, so spans can be logged, counted and passed
around without carrying the value itself.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class PiiEntity(StrEnum):
    CPF = "CPF"
    CNS = "CNS"
    RG = "RG"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    CEP = "CEP"
    ADDRESS = "ADDRESS"
    BIRTH_DATE = "BIRTH_DATE"
    PERSON = "PERSON"
    CRM = "CRM"
    HEALTH_CARD = "HEALTH_CARD"

    @property
    def placeholder_label(self) -> str:
        """Portuguese label used in placeholders such as `[TELEFONE_1]`."""
        return _PLACEHOLDER_LABELS[self]


_PLACEHOLDER_LABELS = {
    PiiEntity.CPF: "CPF",
    PiiEntity.CNS: "CNS",
    PiiEntity.RG: "RG",
    PiiEntity.PHONE: "TELEFONE",
    PiiEntity.EMAIL: "EMAIL",
    PiiEntity.CEP: "CEP",
    PiiEntity.ADDRESS: "ENDERECO",
    PiiEntity.BIRTH_DATE: "DATA_NASC",
    PiiEntity.PERSON: "PESSOA",
    PiiEntity.CRM: "CRM",
    PiiEntity.HEALTH_CARD: "CARTEIRINHA",
}


@dataclass(frozen=True, slots=True)
class PiiSpan:
    entity: PiiEntity
    start: int
    end: int
    score: float
    """Confidence in 0..1; the redactor masks spans at or above its `min_score`."""
    detector: str
    """Name of the rule that matched; never contains the matched value."""

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid span offsets {self.start}..{self.end}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in 0..1, got {self.score}")

    def __len__(self) -> int:
        return self.end - self.start


class PiiDetector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, text: str) -> list[PiiSpan]: ...


class RegexDetector:
    """Detector made of one pattern; subclasses refine matches in `score`.

    The pattern's group `value` (or the whole match when absent) is the span,
    so a context label such as "CPF:" helps matching without being masked.
    """

    entity: PiiEntity
    name: str
    pattern: re.Pattern[str]
    base_score: float = 1.0

    def detect(self, text: str) -> list[PiiSpan]:
        spans = []
        for match in self.pattern.finditer(text):
            group = "value" if "value" in self.pattern.groupindex else 0
            start, end = match.span(group)
            if start == end:
                continue
            score = self.score(match)
            if score is not None:
                spans.append(PiiSpan(self.entity, start, end, score, self.name))
        return spans

    def score(self, match: re.Match[str]) -> float | None:
        """Confidence for a match, or None to discard it."""
        return self.base_score


def only_digits(value: str) -> str:
    return "".join(c for c in value if "0" <= c <= "9")
