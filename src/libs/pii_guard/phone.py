import re

from pii_guard.types import PiiEntity, RegexDetector, only_digits

_MIN_DIGITS_WITH_AREA_CODE = 10
_WITH_AREA_CODE_SCORE = 0.9
_HYPHEN_ONLY_SCORE = 0.7

_OPTIONAL_SPACE = r"[^\S\n]?"
# Area codes have no zero; mobiles have 9 digits starting with 9, landlines 8 starting 2..5.
_SUBSCRIBER = rf"(?:9{_OPTIONAL_SPACE}[0-9]{{4}}|[2-5][0-9]{{3}})"
_PATTERN = re.compile(
    rf"(?<![0-9+])(?P<value>"
    rf"(?:\+?55{_OPTIONAL_SPACE})?(?:\([1-9]{{2}}\)|[1-9]{{2}}){_OPTIONAL_SPACE}{_SUBSCRIBER}"
    rf"{_OPTIONAL_SPACE}-?{_OPTIONAL_SPACE}[0-9]{{4}}"
    rf"|{_SUBSCRIBER}-[0-9]{{4}}"
    rf")(?![0-9])"
)


class PhoneDetector(RegexDetector):
    entity = PiiEntity.PHONE
    name = "phone"
    pattern = _PATTERN

    def score(self, match: re.Match[str]) -> float | None:
        if len(only_digits(match["value"])) >= _MIN_DIGITS_WITH_AREA_CODE:
            return _WITH_AREA_CODE_SCORE

        return _HYPHEN_ONLY_SCORE
