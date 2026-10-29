import re

from pii_guard.types import PiiEntity, RegexDetector

_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"
)


class EmailDetector(RegexDetector):
    entity = PiiEntity.EMAIL
    name = "email"
    pattern = _PATTERN
    base_score = 0.95
