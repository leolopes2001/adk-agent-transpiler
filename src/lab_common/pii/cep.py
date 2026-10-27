"""CEP (postal code): hyphenated format anywhere, plain 8 digits only after a label."""

import re

from lab_common.pii.types import PiiEntity, RegexDetector

_PATTERN = re.compile(
    r"(?P<label>(?i:\bCEP)[^\S\n]*[:.\-]?[^\S\n]*)?"
    r"(?<![0-9])(?P<value>[0-9]{2}\.?[0-9]{3}[^\S\n]?-[^\S\n]?[0-9]{3}|[0-9]{8})(?![0-9])"
)


class CepDetector(RegexDetector):
    entity = PiiEntity.CEP
    name = "cep"
    pattern = _PATTERN

    def score(self, match: re.Match[str]) -> float | None:
        if match["label"]:
            return 0.95
        return 0.8 if "-" in match["value"] else None
