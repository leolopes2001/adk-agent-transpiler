import re

from pii_guard.types import PiiEntity, RegexDetector

_STREET_TYPE = r"(?:Rua|R\.|Av\.?|Avenida|Travessa|Tv\.|Alameda|Al\.|Rodovia|Estrada|Praça)"
_LABELS = "|".join(entity.placeholder_label for entity in PiiEntity)
_ONLY_PLACEHOLDERS = re.compile(rf"(?:[^\S\n]*\[(?:{_LABELS})_\d+\])+[^\S\n]*")
# The rest of the line is masked: number, complement and district identify as much as the street.
_PATTERN = re.compile(
    r"^[^\S\n]*(?P<label>(?i:endere[çc]o(?:[^\S\n]+residencial)?)[^\S\n]*[:\-]?[^\S\n]*)?"
    rf"(?P<value>(?(label)\S[^\n]*?|(?i:{_STREET_TYPE})[^\S\n]+\S[^\n]*?))[^\S\n]*$",
    re.MULTILINE,
)


class AddressDetector(RegexDetector):
    entity = PiiEntity.ADDRESS
    name = "address"
    pattern = _PATTERN
    base_score = 0.8

    def score(self, match: re.Match[str]) -> float | None:
        # What masking leaves behind is not an address, so the redactor's re-scan finds nothing.
        if _ONLY_PLACEHOLDERS.fullmatch(match.group("value")):
            return None

        return self.base_score
