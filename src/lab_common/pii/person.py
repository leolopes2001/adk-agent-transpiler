"""Person names, recognized by the label before them ("Paciente:", "Dr.").

Names have no fixed format, so this detector only looks right after a label
and takes the whole run of capitalized words that follows (no word limit: a
cut-off would leak the rest of a long name). Two safeguards keep exam
names and form fields out of the span:

- the run stops at a field label ("CPF", "Telefone", ...), so "Dr. João CRM 1"
  masks the name and leaves the CRM to its own detector;
- a run made only of medical vocabulary (`medical_terms.txt`: every token of
  the exam catalog plus common clinical words) is not a name, so "Paciente:
  Hemograma Completo" or "Paciente Diabético" are left alone.
"""

import re
from collections.abc import Iterable
from functools import cache
from importlib import resources

from lab_common.pii.types import PiiEntity, PiiSpan
from lab_common.text import tokens

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

_UPPER = "A-ZÀ-ÖØ-Þ"
_LOWER = "a-zß-öø-ÿ"
_WORD = (
    rf"(?:[{_UPPER}][{_LOWER}]+(?:['\u2019\-][{_UPPER}{_LOWER}]+)*"  # Maria, D'Ávila, Ana-Clara
    rf"|[{_UPPER}]{{2,}}(?:['\u2019\-][{_UPPER}]+)*"  # MARIA
    rf"|[{_UPPER}]\.)"  # J.
    # Whole words only: "IgM" or "NS1" must not shrink to "Ig" or "NS", which
    # would no longer match the exclusion list.
    r"(?!\w)"
)
_CONNECTOR = r"(?:d[aeo]s?|D[AEO]S?|e|E)"
_PATTERN = re.compile(
    rf"(?<![\w.])(?i:{_LABEL})(?:(?<=\.)|(?!\w))[^\S\n]*[:\-\u2013]?[^\S\n]*"
    rf"(?P<value>{_WORD}(?:[^\S\n]+(?:{_CONNECTOR}[^\S\n]+)?{_WORD})*)"
)
_WORD_RE = re.compile(rf"{_WORD}|{_CONNECTOR}")

# Form labels (normalized) that end a name when they follow it on the same line.
_FIELD_LABELS = frozenset(
    {
        "cpf",
        "rg",
        "cns",
        "crm",
        "cep",
        "tel",
        "telefone",
        "celular",
        "data",
        "nascimento",
        "dn",
        "idade",
        "sexo",
        "convenio",
        "carteirinha",
        "matricula",
        "endereco",
        "email",
        "assinatura",
    }
)


@cache
def medical_terms() -> frozenset[str]:
    """Normalized tokens that never make up a person name on their own."""
    content = resources.files(__package__).joinpath("medical_terms.txt").read_text("utf-8")
    terms: set[str] = set()
    for line in content.splitlines():
        terms.update(tokens(line.split("#", 1)[0]))
    return frozenset(terms)


class PersonDetector:
    entity = PiiEntity.PERSON
    name = "person"

    def __init__(self, extra_terms: Iterable[str] = ()) -> None:
        self._excluded = medical_terms() | {t for term in extra_terms for t in tokens(term)}

    def detect(self, text: str) -> list[PiiSpan]:
        spans = []
        for match in _PATTERN.finditer(text):
            role = next(r for r in _ROLE_LABELS if match[r] is not None)
            span = self._name_span(match, role)
            if span is not None:
                spans.append(span)
        return spans

    def _name_span(self, match: re.Match[str], role: str) -> PiiSpan | None:
        kept: list[re.Match[str]] = []
        for word in _WORD_RE.finditer(match["value"]):
            if _FIELD_LABELS.intersection(tokens(word[0])):
                break
            kept.append(word)
        while kept and _is_connector(kept[-1][0]):
            kept.pop()

        name_tokens = tokens(" ".join(w[0] for w in kept))
        if not name_tokens or all(t in self._excluded for t in name_tokens):
            return None
        names = sum(not _is_connector(w[0]) for w in kept)
        offset = match.start("value")
        return PiiSpan(
            PiiEntity.PERSON,
            offset + kept[0].start(),
            offset + kept[-1].end(),
            FULL_NAME_SCORE if names > 1 else SINGLE_WORD_SCORE,
            f"person.{role}",
        )


def _is_connector(word: str) -> bool:
    return re.fullmatch(_CONNECTOR, word) is not None
