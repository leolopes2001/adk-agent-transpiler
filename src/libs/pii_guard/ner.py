"""Person names found by a local spaCy model through Presidio: no text leaves the process."""

import re
from functools import cache
from typing import Any

from pii_guard.labels import PERSONAL_LABELS
from pii_guard.person import medical_terms
from pii_guard.text import LETTERS_PER_EDIT, edit_distance, tokens
from pii_guard.types import PiiEntity, PiiSpan

MODEL = "pt_core_news_md"
LANGUAGE = "pt"
SCORE = 0.55
"""Below every rule (0.6 and up): on an overlap the rule's span is kept."""

_WORD = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*")
# A name holds only words, spaces, apostrophes and hyphens; anything else ends the phrase.
_PHRASE = re.compile(r"[\w'\u2019\-]+(?:[^\S\n]+[\w'\u2019\-]+)*")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_[0-9]+\]")
_CONNECTORS = frozenset({"de", "da", "do", "das", "dos", "e"})


class NerPersonDetector:
    name = "ner.person"

    def __init__(self, analyzer: Any | None = None) -> None:
        self._analyzer = analyzer if analyzer is not None else _analyzer()
        self._non_name = medical_terms() | PERSONAL_LABELS

    def detect(self, text: str) -> list[PiiSpan]:
        spans: list[PiiSpan] = []
        # The model reads a name differently with and without the lines around it: both count.
        for offset, unit in _units(text):
            titled = title_case(unit)
            # The model misses upper-case names; the title-case copy has the same offsets.
            for variant in (unit,) if titled == unit else (unit, titled):
                for found in self._analyzer.analyze(
                    variant, language=LANGUAGE, entities=["PERSON"]
                ):
                    found_spans = self._name_spans(text, offset + found.start, offset + found.end)
                    spans.extend(
                        s for s in found_spans if variant is unit or _upper_or_full(text, s)
                    )

        return spans

    def _name_spans(self, text: str, start: int, end: int) -> list[PiiSpan]:
        # The model can stop inside a word ("NS" of "NS1"); whole words decide.
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1

        placeholders = [m.span() for m in _PLACEHOLDER.finditer(text)]
        spans = []
        for phrase in _PHRASE.finditer(text, start, end):
            if any(p_start <= phrase.start() < p_end for p_start, p_end in placeholders):
                continue

            words = [w for w in _WORD.finditer(phrase[0]) if w[0].casefold() not in _CONNECTORS]
            # Medical words at the edges are trimmed; inside a name they stay masked.
            while words and not self._is_name_word(words[0][0]):
                words.pop(0)
            while words and not self._is_name_word(words[-1][0]):
                words.pop()
            if words:
                offset = phrase.start()
                spans.append(
                    PiiSpan(
                        PiiEntity.PERSON,
                        offset + words[0].start(),
                        offset + words[-1].end(),
                        SCORE,
                        self.name,
                    )
                )

        return spans

    def _is_name_word(self, word: str) -> bool:
        if any(c.isdigit() for c in word):
            return False

        word_tokens = tokens(word)
        return bool(word_tokens) and not all(self._is_non_name(t) for t in word_tokens)

    # Fuzzy as well: OCR misspells exams ("FERRLTINA") the way it misspells labels.
    def _is_non_name(self, token: str) -> bool:
        if token in self._non_name:
            return True

        max_edits = len(token) // LETTERS_PER_EDIT
        return max_edits > 0 and any(
            abs(len(term) - len(token)) <= max_edits and edit_distance(token, term) <= max_edits
            for term in self._non_name
        )


@cache
def shared_ner_detector() -> NerPersonDetector:
    """One model per process: loading takes about a second."""
    return NerPersonDetector()


def _units(text: str) -> list[tuple[int, str]]:
    units = [(0, text)]
    if "\n" in text:
        units += [(m.start(), m[0]) for m in re.finditer(r"[^\n]*\S[^\n]*", text)]

    return units


# On the title-case copy one lower-case word is too weak ("(fictício)"); "gustavo nunes" is not.
def _upper_or_full(text: str, span: PiiSpan) -> bool:
    value = text[span.start : span.end]
    return value.isupper() or len(_WORD.findall(value)) > 1


def title_case(text: str) -> str:
    chars = []
    after_letter = False
    for char in text:
        chars.append(char.lower() if after_letter else char.upper())
        after_letter = char.isalpha()

    titled = "".join(chars)
    return titled if len(titled) == len(text) else text


def _analyzer() -> Any:
    # Imported here: only the OCR server and the agent install Presidio and spaCy.
    import spacy  # noqa: PLC0415
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry  # noqa: PLC0415
    from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: PLC0415
    from presidio_analyzer.predefined_recognizers import SpacyRecognizer  # noqa: PLC0415

    # Presidio would download a missing model at run time; it is installed at build time.
    if not spacy.util.is_package(MODEL):
        raise RuntimeError(f"spaCy model {MODEL} is not installed")

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": MODEL}],
            # Only person names: the model's places, organizations and MISC are not requested.
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {"PER": "PERSON"},
                "labels_to_ignore": ["LOC", "ORG", "MISC"],
            },
        }
    )
    # Only the model's recognizer: Brazilian identifiers stay with the rules and their validators.
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    registry.add_recognizer(
        SpacyRecognizer(supported_language=LANGUAGE, supported_entities=["PERSON"])
    )
    return AnalyzerEngine(
        registry=registry, nlp_engine=provider.create_engine(), supported_languages=[LANGUAGE]
    )
