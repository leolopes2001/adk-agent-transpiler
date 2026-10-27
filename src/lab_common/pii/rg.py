"""RG (registro geral), only next to a label: the number alone has no fixed
format across states and would collide with any other document number."""

import re

from lab_common.pii.types import PiiEntity, RegexDetector

_PATTERN = re.compile(
    r"(?i:\b(?:RG|R\.G\.|identidade|carteira\s+de\s+identidade))"
    r"[^\S\n]*(?i:n[º°o.]*)?[^\S\n]*[:\-]?[^\S\n]*"
    r"(?P<value>(?:[A-Z]{2}[^\S\n]?-?[^\S\n]?)?[0-9]{1,2}\.?[0-9]{3}\.?[0-9]{3}(?:-?[0-9Xx])?)"
    r"(?![0-9])"
)


class RgDetector(RegexDetector):
    entity = PiiEntity.RG
    name = "rg"
    pattern = _PATTERN
    base_score = 0.85
