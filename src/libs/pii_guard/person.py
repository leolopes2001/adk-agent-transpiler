import re
from collections.abc import Iterable
from functools import cache
from importlib import resources

from pii_guard.labels import LABELS_ENDING_NAME, PERSONAL_LABELS, SECTION_TITLES
from pii_guard.text import tokens
from pii_guard.types import PiiEntity, PiiSpan

SINGLE_WORD_SCORE = 0.6
FULL_NAME_SCORE = 0.85

_ROLE_LABELS = {
    "patient": (
        r"paciente|nome\s+d[oa]\s+paciente|nome\s+completo|nome|benefici[aá]ri[oa]"
        r"|sr\.|sra\.|srta\."
    ),
    "professional": (
        r"dr\.|dra\.|dr\(a\)\.|m[eé]dic[oa](?:\(a\))?(?:\s+(?:solicitante|respons[aá]vel))?"
        r"|solicitante|profissional"
    ),
    "other": r"respons[aá]vel|acompanhante|nome\s+d[ao]\s+(?:m[aã]e|pai)|m[aã]e|pai",
}
_LABEL = "|".join(f"(?P<{role}>{labels})" for role, labels in _ROLE_LABELS.items())

_UPPER_LETTERS = "A-ZÀ-ÖØ-Þ"
_LOWER_LETTERS = "a-zß-öø-ÿ"
_TITLE_CASE_WORD = (
    rf"[{_UPPER_LETTERS}][{_LOWER_LETTERS}]+(?:['\u2019\-][{_UPPER_LETTERS}{_LOWER_LETTERS}]+)*"
)
_ALL_CAPS_WORD = rf"[{_UPPER_LETTERS}]{{2,}}(?:['\u2019\-][{_UPPER_LETTERS}]+)*"
_INITIAL = rf"[{_UPPER_LETTERS}]\."
# Whole words only: "IgM" or "NS1" must not shrink to "Ig" or "NS" and escape the exclusion list.
_WORD = rf"(?:{_TITLE_CASE_WORD}|{_ALL_CAPS_WORD}|{_INITIAL})(?!\w)"
_CONNECTOR = r"(?:d[aeo]s?|D[AEO]S?|e|E)"
# No word limit: cutting a long name short would leak the rest of it.
_CAPITALIZED_NAME = re.compile(
    rf"(?<![\w.])(?i:{_LABEL})(?:(?<=\.)|(?!\w))[^\S\n]*[:\-\u2013]?[^\S\n]*"
    rf"(?P<value>{_WORD}(?:[^\S\n]+(?:{_CONNECTOR}[^\S\n]+)?{_WORD})*)"
)
_CAPITALIZED_WORD = re.compile(rf"{_WORD}|{_CONNECTOR}")

_ANY_CASE_WORD = r"[^\W\d_]+(?:['\u2019\-][^\W\d_]+)*(?!\w)\.?"
# After an explicit separator the name may be in any case, or on the next line.
_LABELED_NAME = re.compile(
    rf"(?<![\w.])(?i:{_LABEL})(?:(?<=\.)[^\S\n]*:?|(?!\w)[^\S\n]*[:\-\u2013])"
    rf"[^\S\n]*(?:\n[^\S\n]*)?"
    rf"(?P<value>{_ANY_CASE_WORD}(?:[^\S\n]+{_ANY_CASE_WORD})*)"
)
_ANY_CASE_WORD_RE = re.compile(_ANY_CASE_WORD)

_NAME_PATTERNS = ((_CAPITALIZED_NAME, _CAPITALIZED_WORD), (_LABELED_NAME, _ANY_CASE_WORD_RE))
# Before a colon only these end the name: "Dr. João Exemplo: Ferritina" keeps "Exemplo".
_LABELS_BEFORE_COLON = PERSONAL_LABELS | SECTION_TITLES


class PersonDetector:
    entity = PiiEntity.PERSON
    name = "person"

    def __init__(self, extra_terms: Iterable[str] = ()) -> None:
        self._non_name_terms = medical_terms() | {t for term in extra_terms for t in tokens(term)}

    def detect(self, text: str) -> list[PiiSpan]:
        found = []
        for pattern, words in _NAME_PATTERNS:
            for match in pattern.finditer(text):
                role = next(r for r in _ROLE_LABELS if match[r] is not None)
                span = self._name_span(match, role, words)
                if span is not None:
                    found.append(span)

        return _without_overlaps(found)

    def _name_span(self, match: re.Match[str], role: str, words: re.Pattern[str]) -> PiiSpan | None:
        kept = _name_words(match, words)
        name_tokens = tokens(" ".join(w[0] for w in kept))
        # A run of medical vocabulary ("Paciente: Hemograma Completo") is not a name.
        if not name_tokens or all(t in self._non_name_terms for t in name_tokens):
            return None

        offset = match.start("value")
        names = sum(not _is_connector(w[0]) for w in kept)
        return PiiSpan(
            PiiEntity.PERSON,
            offset + kept[0].start(),
            offset + kept[-1].end(),
            FULL_NAME_SCORE if names > 1 else SINGLE_WORD_SCORE,
            f"person.{role}",
        )


_CATALOG_SECTION = "# Every non-numeric token"


@cache
def medical_terms() -> frozenset[str]:
    return frozenset().union(*_term_sections().values())


@cache
def catalog_terms() -> frozenset[str]:
    sections = _term_sections()
    return next(terms for title, terms in sections.items() if title.startswith(_CATALOG_SECTION))


# Sections are separated by a line holding only "#" and named by their first comment.
@cache
def _term_sections() -> dict[str, frozenset[str]]:
    content = resources.files(__package__).joinpath("medical_terms.txt").read_text("utf-8")
    sections: dict[str, set[str]] = {}
    title = ""
    for line in content.splitlines():
        if line.strip() == "#":
            title = ""
        elif line.startswith("#"):
            title = title or line
        else:
            sections.setdefault(title, set()).update(tokens(line.split("#", 1)[0]))

    return {title: frozenset(terms) for title, terms in sections.items()}


def _name_words(match: re.Match[str], words: re.Pattern[str]) -> list[re.Match[str]]:
    offset = match.start("value")
    kept: list[re.Match[str]] = []
    for word in words.finditer(match["value"]):
        word_tokens = tokens(word[0])
        followed_by_colon = match.string.startswith(":", offset + word.end())
        if LABELS_ENDING_NAME.intersection(word_tokens) or (
            followed_by_colon and _LABELS_BEFORE_COLON.intersection(word_tokens)
        ):
            break

        kept.append(word)

    while kept and _is_connector(kept[-1][0]):
        kept.pop()

    return kept


def _without_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    kept: list[PiiSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, -len(s))):
        if not kept or span.start >= kept[-1].end:
            kept.append(span)

    return kept


def _is_connector(word: str) -> bool:
    return re.fullmatch(_CONNECTOR, word, re.IGNORECASE) is not None
