import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

_UUID = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HEX_LETTERS = frozenset("abcdefABCDEF")


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


# Offsets only, never the value, so spans can be logged, counted and passed around.
@dataclass(frozen=True, slots=True)
class PiiSpan:
    entity: PiiEntity
    start: int
    end: int
    score: float
    detector: str

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
    entity: PiiEntity
    name: str
    pattern: re.Pattern[str]
    base_score: float = 1.0

    def detect(self, text: str) -> list[PiiSpan]:
        # The `value` group lets a label such as "CPF:" help matching without being masked.
        group = "value" if "value" in self.pattern.groupindex else 0
        spans = []
        for match in self.pattern.finditer(text):
            start, end = match.span(group)
            if start == end or inside_hex_token(text, start, end):
                continue

            score = self.score(match)
            if score is not None:
                spans.append(PiiSpan(self.entity, start, end, score, self.name))

        return spans

    def score(self, match: re.Match[str]) -> float | None:
        return self.base_score


def inside_hex_token(text: str, start: int, end: int) -> bool:
    if _overlaps_uuid(text, start, end):
        return True

    return any(_touches_hex_run(text, edge, step) for edge, step in ((start - 1, -1), (end, 1)))


def only_digits(value: str) -> str:
    return "".join(c for c in value if "0" <= c <= "9")


def _overlaps_uuid(text: str, start: int, end: int) -> bool:
    return any(m.start() < end and start < m.end() for m in _UUID.finditer(text))


def _touches_hex_run(text: str, edge: int, step: int) -> bool:
    if _is_hex_run(_ascii_alnum_run(text, edge, step), needs_letter=False):
        return True

    # Across a hyphen the run needs a hex letter, so "31-90000-0000" is still a phone.
    hyphen = 0 <= edge < len(text) and text[edge] == "-"
    return hyphen and _is_hex_run(_ascii_alnum_run(text, edge + step, step), needs_letter=True)


def _ascii_alnum_run(text: str, index: int, step: int) -> str:
    chars = []
    while 0 <= index < len(text) and text[index].isascii() and text[index].isalnum():
        chars.append(text[index])
        index += step

    return "".join(chars)


def _is_hex_run(run: str, *, needs_letter: bool) -> bool:
    chars = set(run)
    if not chars or not chars <= _HEX_DIGITS:
        return False

    return not needs_letter or bool(chars & _HEX_LETTERS)
