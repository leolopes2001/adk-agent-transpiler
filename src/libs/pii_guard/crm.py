import re

from pii_guard.types import PiiEntity, RegexDetector

_PATTERN = re.compile(
    r"\b(?P<value>(?i:CRM)[^\S\n]*[-/:]?[^\S\n]*(?:[A-Z]{2}[^\S\n]*[-/:]?[^\S\n]*)?"
    r"(?:n[º°o.]*[^\S\n]*)?[0-9][0-9.]{2,8}[0-9])(?![0-9])"
)


class CrmDetector(RegexDetector):
    entity = PiiEntity.CRM
    name = "crm"
    pattern = _PATTERN
    base_score = 0.95
