import re

from lab_common.pii.types import PiiEntity, RegexDetector, only_digits

MIN_DIGITS = 5

_LABEL = (
    r"carteirinha|carteira[^\S\n]+do[^\S\n]+(?:plano|conv[eê]nio)|matr[ií]cula"
    r"|conv[eê]nio|cart[aã]o[^\S\n]+(?:do[^\S\n]+)?(?:plano|conv[eê]nio|benefici[aá]rio)"
)
# A plan name may sit between label and number ("Convênio: Plano X nº").
_PATTERN = re.compile(
    rf"(?i:\b(?:{_LABEL}))\b[^\n0-9]{{0,40}}?"
    r"(?P<value>[0-9](?:[0-9.\-/]|[^\S\n](?=[0-9]))*[0-9])(?![0-9])"
)


class HealthCardDetector(RegexDetector):
    entity = PiiEntity.HEALTH_CARD
    name = "health_card"
    pattern = _PATTERN
    base_score = 0.9

    def score(self, match: re.Match[str]) -> float | None:
        return self.base_score if len(only_digits(match["value"])) >= MIN_DIGITS else None
