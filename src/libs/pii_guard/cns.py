import re

from pii_guard.types import PiiEntity, RegexDetector, only_digits

CNS_LENGTH = 15
# Definitive (1, 2) and provisional (7, 8, 9) cards share the same mod-11 check.
_CARD_TYPE_DIGITS = (1, 2, 7, 8, 9)

_PATTERN = re.compile(
    r"(?<![0-9])(?P<value>[1-27-9][0-9]{2}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})(?![0-9])"
)


def is_valid_cns(value: str) -> bool:
    digits = [int(c) for c in only_digits(value)]
    if len(digits) != CNS_LENGTH or digits[0] not in _CARD_TYPE_DIGITS:
        return False

    return sum(d * w for d, w in zip(digits, range(CNS_LENGTH, 0, -1), strict=True)) % 11 == 0


class CnsDetector(RegexDetector):
    entity = PiiEntity.CNS
    name = "cns"
    pattern = _PATTERN

    def score(self, match: re.Match[str]) -> float | None:
        value = match["value"]
        return 1.0 if is_valid_cns(value) else None
