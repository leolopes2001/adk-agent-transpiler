import re

from pii_guard.types import PiiEntity, RegexDetector

_MONTHS = (
    r"(?:janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro"
    r"|novembro|dezembro)"
)
_DATE = (
    r"[0-3]?[0-9][^\S\n]?[/.\-][^\S\n]?[01]?[0-9][^\S\n]?[/.\-][^\S\n]?(?:[0-9]{4}|[0-9]{2})"
    rf"|[0-3]?[0-9][^\S\n]+de[^\S\n]+{_MONTHS}[^\S\n]+de[^\S\n]+[0-9]{{4}}"
)
_PATTERN = re.compile(
    r"(?i:\b(?:data[^\S\n]+de[^\S\n]+nascimento|nascid[oa](?:[^\S\n]+em)?|nasc\w*|d\.?[^\S\n]?n\.)"
    r"|\bDN\b)"
    rf"\.?[^\S\n]*[:\-]?[^\S\n]*(?P<value>(?i:{_DATE}))(?![0-9])"
)


class BirthDateDetector(RegexDetector):
    entity = PiiEntity.BIRTH_DATE
    name = "birth_date"
    pattern = _PATTERN
    base_score = 0.9
