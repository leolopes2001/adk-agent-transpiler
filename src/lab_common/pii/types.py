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


_UUID = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")
_HEX = frozenset("0123456789abcdefABCDEF")
_HEX_LETTERS = frozenset("abcdefABCDEF")


def _alnum_run(text: str, index: int, step: int) -> str:
    """Letters and digits read from `index` towards `step` (-1 left, +1 right)."""
    chars = []
    while 0 <= index < len(text) and text[index].isascii() and text[index].isalnum():
        chars.append(text[index])
        index += step
    return "".join(chars)


def _hex_token(run: str, *, needs_letter: bool) -> bool:
    return bool(run) and set(run) <= _HEX and (not needs_letter or bool(set(run) & _HEX_LETTERS))


def inside_hex_token(text: str, start: int, end: int) -> bool:
    """Whether `text[start:end]` is a piece of a UUID, hash or other hex id.

    True when the span overlaps a UUID, is glued to a run made only of hex
    characters, or is joined by a hyphen to a hex run with at least one letter
    ("2f3b5fd6-2866-4383-..."). A label glued to a number ("CPF20239361369",
    "Telefone31900000000") or a hyphenated area code ("31-90000-0000") does
    not count: those runs are not hex only, or have no letter.
    """
    if any(m.start() < end and start < m.end() for m in _UUID.finditer(text)):
        return True
    for edge, step in ((start - 1, -1), (end, 1)):
        if _hex_token(_alnum_run(text, edge, step), needs_letter=False):
            return True
        hyphen = 0 <= edge < len(text) and text[edge] == "-"
        if hyphen and _hex_token(_alnum_run(text, edge + step, step), needs_letter=True):
            return True
    return False


class RegexDetector:
    """Detector made of one pattern; subclasses refine matches in `score`.

    The pattern's group `value` (or the whole match when absent) is the span,
    so a context label such as "CPF:" helps matching without being masked.
    Matches inside a hex token (UUIDs, hashes, ids in logs) are dropped: a run
    of digits there is not a document number or a phone.
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
            if start == end or inside_hex_token(text, start, end):
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
