import re
import unicodedata

from pii_guard.types import PiiEntity, PiiSpan

SCORE = 1.0
"""Wins every overlap, so no part of the value is left to a narrower detector."""
MAX_LABEL_WORDS = 4

# Short labels are one edit away from exam acronyms ("CPK"), so only known misreadings count.
_EXACT_LABELS = {
    "CPF": PiiEntity.CPF,
    "CPE": PiiEntity.CPF,
    "CPR": PiiEntity.CPF,
    "TEL": PiiEntity.PHONE,
    "NASC": PiiEntity.BIRTH_DATE,
    "RG": PiiEntity.RG,
    "CNS": PiiEntity.CNS,
}
_FUZZY_LABELS = {
    "TELEFONE": PiiEntity.PHONE,
    "DATADENASCIMENTO": PiiEntity.BIRTH_DATE,
    "NASCIMENTO": PiiEntity.BIRTH_DATE,
    "CARTEIRINHA": PiiEntity.HEALTH_CARD,
}
# One edit per six letters: "JELEFONE" and "PATA DE NAScIMENTO" match, "CRESCIMENTO" does not.
_LETTERS_PER_EDIT = 6
_DIGITS_READ_AS_LETTERS = str.maketrans("015", "OIS")

_LINE = re.compile(r"[^\n]+")
# "[" and "_" keep placeholders such as [TELEFONE_1] from reading as a label.
_LINE_PREFIX = re.compile(r"^[^\w\[]*")
_WORD = re.compile(r"[^\W_]+(?!\w)")
_WORD_GAP = re.compile(r"[^\S\n]+")
_SEPARATORS = " \t:.,;=|!-\u2013\u2014"
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_[0-9]+\]")


class LabeledFieldDetector:
    """Masks what follows a personal-data label at the start of a line, even unreadable.

    OCR misreads the label and the value alike ("CPE: 23.456. 89-09"), and a value
    that no longer looks like a document would otherwise pass unmasked.
    """

    name = "labeled_field"

    def detect(self, text: str) -> list[PiiSpan]:
        spans = []
        for line in _LINE.finditer(text):
            found = _labeled_value(line[0])
            if found is not None:
                entity, start, end = found
                offset = line.start()
                spans.append(PiiSpan(entity, offset + start, offset + end, SCORE, self.name))

        return spans


def _labeled_value(line: str) -> tuple[PiiEntity, int, int] | None:
    label = _label_at_start(line)
    if label is None:
        return None

    entity, label_end = label
    value = line[label_end:].lstrip(_SEPARATORS)
    start, end = len(line) - len(value), len(line.rstrip())
    if start >= end or not any(c.isalnum() for c in _PLACEHOLDER.sub("", line[start:end])):
        return None

    return entity, start, end


def _label_at_start(line: str) -> tuple[PiiEntity, int] | None:
    position = len(line) - len(_LINE_PREFIX.sub("", line, count=1))
    words: list[str] = []
    while len(words) < MAX_LABEL_WORDS and (word := _WORD.match(line, position)):
        words.append(_letters(word[0]))
        entity = _label_entity("".join(words), single_word=len(words) == 1)
        if entity is not None:
            return entity, word.end()

        gap = _WORD_GAP.match(line, word.end())
        if gap is None:
            break
        position = gap.end()

    return None


def _label_entity(candidate: str, *, single_word: bool) -> PiiEntity | None:
    if single_word and candidate in _EXACT_LABELS:
        return _EXACT_LABELS[candidate]

    for label, entity in _FUZZY_LABELS.items():
        max_edits = len(label) // _LETTERS_PER_EDIT
        if abs(len(candidate) - len(label)) <= max_edits and (
            _edit_distance(candidate, label) <= max_edits
        ):
            return entity

    return None


def _letters(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.upper())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.translate(_DIGITS_READ_AS_LETTERS)


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current

    return previous[-1]
