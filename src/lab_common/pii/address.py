"""Street addresses: a line that starts with a street type or an address label.

The rest of the line is masked, since number, complement and district are as
identifying as the street name.
"""

import re

from lab_common.pii.types import PiiEntity, RegexDetector

_STREET = r"(?:Rua|R\.|Av\.?|Avenida|Travessa|Tv\.|Alameda|Al\.|Rodovia|Estrada|Praça)"
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
