"""CPF (Cadastro de Pessoas Físicas), with or without punctuation."""

import re

from lab_common.pii.types import PiiEntity, RegexDetector, only_digits

CPF_LENGTH = 11
VALID_SCORE = 1.0
FORMAT_ONLY_SCORE = 0.6
"""A CPF-shaped number with wrong check digits is still masked by default: OCR
often misreads one digit, and the value is still the patient's identifier."""

_PATTERN = re.compile(
    r"(?<![0-9])(?P<value>[0-9]{3}[.\s]?[0-9]{3}[.\s]?[0-9]{3}\s?[-.]?\s?[0-9]{2})(?![0-9])"
)


def is_valid_cpf(value: str) -> bool:
    digits = [int(c) for c in only_digits(value)]
    if len(digits) != CPF_LENGTH or len(set(digits)) == 1:
        return False
    for size in (9, 10):
        total = sum(d * w for d, w in zip(digits[:size], range(size + 1, 1, -1), strict=True))
        if (total * 10 % 11) % 10 != digits[size]:
            return False
    return True


class CpfDetector(RegexDetector):
    entity = PiiEntity.CPF
    pattern = _PATTERN

    def __init__(self, *, require_valid: bool = False) -> None:
        self._require_valid = require_valid
        self.name = "cpf.valid" if require_valid else "cpf"

    def score(self, match: re.Match[str]) -> float | None:
        value = match["value"]
        if is_valid_cpf(value):
            return VALID_SCORE
        return None if self._require_valid else FORMAT_ONLY_SCORE
