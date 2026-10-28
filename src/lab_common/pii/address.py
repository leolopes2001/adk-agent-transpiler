"""Street addresses: a line that starts with a street type or an address label.

The rest of the line is masked, since number, complement and district are as
identifying as the street name. A labeled value made only of placeholders
("Endereço: [ENDERECO_1]", what masking leaves behind) is not an address, so
the redactor's re-scan of its own output finds nothing there.
"""

import re

from lab_common.pii.types import PiiEntity, RegexDetector

_STREET = r"(?:Rua|R\.|Av\.?|Avenida|Travessa|Tv\.|Alameda|Al\.|Rodovia|Estrada|Praça)"
_LABELS = "|".join(entity.placeholder_label for entity in PiiEntity)
_PLACEHOLDERS = re.compile(rf"(?:[^\S\n]*\[(?:{_LABELS})_\d+\])+[^\S\n]*")
_PATTERN = re.compile(
    r"^[^\S\n]*(?P<label>(?i:endere[çc]o(?:[^\S\n]+residencial)?)[^\S\n]*[:\-]?[^\S\n]*)?"
    rf"(?P<value>(?(label)\S[^\n]*?|(?i:{_STREET})[^\S\n]+\S[^\n]*?))[^\S\n]*$",
    re.MULTILINE,
)


class AddressDetector(RegexDetector):
    entity = PiiEntity.ADDRESS
    name = "address"
    pattern = _PATTERN
    base_score = 0.8

    def score(self, match: re.Match[str]) -> float | None:
        if _PLACEHOLDERS.fullmatch(match.group("value")):
            return None
        return self.base_score
