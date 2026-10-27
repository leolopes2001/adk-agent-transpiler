"""Brazilian phone numbers: optional +55, optional area code, 8 or 9 digits."""

import re

from lab_common.pii.types import PiiEntity, RegexDetector, only_digits

# Area code plus subscriber number.
_MIN_FULL_DIGITS = 10

_SEP = r"[^\S\n]?"
# Area codes have no zero digit; mobile numbers have 9 digits starting with 9,
# landlines 8 digits starting with 2..5.
_SUBSCRIBER = rf"(?:9{_SEP}[0-9]{{4}}|[2-5][0-9]{{3}})"
_PATTERN = re.compile(
    rf"(?<![0-9+])(?P<value>"
    rf"(?:\+?55{_SEP})?(?:\([1-9]{{2}}\)|[1-9]{{2}}){_SEP}{_SUBSCRIBER}{_SEP}-?{_SEP}[0-9]{{4}}"
    rf"|{_SUBSCRIBER}-[0-9]{{4}}"
    rf")(?![0-9])"
)


class PhoneDetector(RegexDetector):
    entity = PiiEntity.PHONE
    name = "phone"
    pattern = _PATTERN

    def score(self, match: re.Match[str]) -> float | None:
        # Without area code the match relies on the hyphen alone.
        return 0.9 if len(only_digits(match["value"])) >= _MIN_FULL_DIGITS else 0.7
