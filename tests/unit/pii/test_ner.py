"""The NER model on the measurement corpus (README, "Segurança"): names it adds to the rules,
exams it must not take for names, and what still escapes."""

import json
from pathlib import Path
from typing import Any

import pytest

from pii_guard import (
    NerPersonDetector,
    PiiEngineError,
    PiiEntity,
    PiiRedactor,
    default_detectors,
    document_detectors,
    shared_ner_detector,
)
from pii_guard.cpf import FORMAT_ONLY_SCORE
from pii_guard.ner import SCORE, title_case
from pii_guard.person import SINGLE_WORD_SCORE
from pii_guard.redactor import detect_all

pytestmark = pytest.mark.unit

CASES_FILE = Path(__file__).resolve().parents[2] / "data" / "pii_ner_cases.jsonl"
CASES = [json.loads(line) for line in CASES_FILE.read_text("utf-8").splitlines() if line]


def _ids(case: dict[str, Any]) -> str:
    return str(case["id"])


def _params(*, escapes: bool) -> list[Any]:
    return [
        pytest.param(case, id=_ids(case))
        for case in CASES
        if case["pii"] and case.get("escapes", False) == escapes
    ]


@pytest.fixture(scope="module")
def redactor() -> PiiRedactor:
    return PiiRedactor(document_detectors())


def _unmasked(redactor: PiiRedactor, text: str, value: str) -> str:
    """The letters and digits of `value` that no span covers."""
    start = text.index(value)
    spans = redactor.redact(text).spans
    return "".join(
        text[i]
        for i in range(start, start + len(value))
        if text[i].isalnum() and not any(s.start <= i < s.end for s in spans)
    )


def test_corpus_has_every_kind_of_case() -> None:
    entities = {pii["entity"] for case in CASES for pii in case["pii"]}

    assert len(CASES) >= 50
    assert {"PERSON", "CPF", "EMAIL", "PHONE", "ADDRESS"} <= entities
    assert sum(bool(case["medical"]) for case in CASES) >= 15


@pytest.mark.parametrize("case", _params(escapes=False))
def test_personal_data_is_masked_whole(redactor: PiiRedactor, case: dict[str, Any]) -> None:
    for pii in case["pii"]:
        assert _unmasked(redactor, case["text"], pii["value"]) == "", pii["entity"]


@pytest.mark.parametrize("case", [pytest.param(c, id=_ids(c)) for c in CASES if c["medical"]])
def test_exam_terms_are_never_masked(redactor: PiiRedactor, case: dict[str, Any]) -> None:
    masked = redactor.redact(case["text"]).text

    for term in case["medical"]:
        assert term in masked


@pytest.mark.parametrize("case", [pytest.param(c, id=_ids(c)) for c in CASES])
def test_masking_leaves_nothing_behind(redactor: PiiRedactor, case: dict[str, Any]) -> None:
    assert not redactor.redact(case["text"]).has_residual


# Documented in the README: places without a label still escape (LOCATION is not used).
@pytest.mark.parametrize("case", _params(escapes=True))
@pytest.mark.xfail(strict=True, reason="known escape, documented in the README")
def test_known_escapes(redactor: PiiRedactor, case: dict[str, Any]) -> None:
    for pii in case["pii"]:
        assert _unmasked(redactor, case["text"], pii["value"]) == ""


def test_ner_adds_names_the_rules_miss() -> None:
    rules = PiiRedactor(default_detectors())
    names = [
        (case["text"], pii["value"])
        for case in CASES
        for pii in case["pii"]
        if pii["entity"] == "PERSON" and not case.get("escapes")
    ]
    with_ner = PiiRedactor(document_detectors())

    missed_by_rules = [v for text, v in names if _unmasked(rules, text, v)]
    missed_with_ner = [v for text, v in names if _unmasked(with_ner, text, v)]
    assert len(missed_by_rules) >= 15
    assert missed_with_ner == []


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("MARIA FICTÍCIA DA SILVA - 42 anos", "MARIA FICTÍCIA DA SILVA"),
        ("FERNANDA ROCHA ALVES", "FERNANDA ROCHA ALVES"),
        ("DR ROBERTO", "ROBERTO"),
    ],
)
def test_upper_case_names_are_found_through_the_title_case_copy(text: str, name: str) -> None:
    spans = shared_ner_detector().detect(text)

    assert name in [text[s.start : s.end] for s in spans]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Maria Fictícia da Silva - Hemograma completo", "Hemograma completo"),
        ("Solicitado por Dr. João Exemplo: Ferritina e Ferro sérico", "Ferritina e Ferro sérico"),
        ("Exames: Dengue NS1, Hemograma completo", "Dengue NS1"),
        ("Convênio: Plano Exemplo Saúde (fictício)", "(fictício)"),
    ],
)
def test_spans_stop_at_exam_words_digits_and_weak_title_case_hits(text: str, kept: str) -> None:
    masked = PiiRedactor(document_detectors()).redact(text).text

    assert kept in masked


def test_placeholders_are_not_names() -> None:
    text = "[PESSOA_1]\nPaciente: [PESSOA_2] - 42 anos\nDR [PESSOA_3]"

    assert shared_ner_detector().detect(text) == []


def test_ner_spans_lose_every_overlap_with_a_rule() -> None:
    text = "Endereço: Rua Maria Fictícia, 123 - Centro"
    spans = detect_all(document_detectors(), text, 0.5)

    assert [(s.entity, text[s.start : s.end]) for s in spans] == [
        (PiiEntity.ADDRESS, "Rua Maria Fictícia, 123 - Centro")
    ]
    assert min(SINGLE_WORD_SCORE, FORMAT_ONLY_SCORE) > SCORE


def test_title_case_keeps_offsets() -> None:
    text = "MARIA DA SILVA - ÁCIDO ÚRICO\nfernanda o'neil"

    titled = title_case(text)

    assert titled == "Maria Da Silva - Ácido Úrico\nFernanda O'Neil"
    assert len(titled) == len(text)


def test_the_model_is_loaded_once_per_process() -> None:
    assert shared_ner_detector() is shared_ner_detector()
    assert any(d is shared_ner_detector() for d in document_detectors())


class _BrokenAnalyzer:
    def analyze(self, text: str, **_: Any) -> list[Any]:
        raise RuntimeError(f"spaCy failed on {text!r}")


def test_a_model_failure_fails_closed_without_quoting_the_text() -> None:
    redactor = PiiRedactor([*default_detectors(), NerPersonDetector(_BrokenAnalyzer())])

    with pytest.raises(PiiEngineError) as raised:
        redactor.redact("Paciente: Maria Fictícia da Silva")

    assert "Maria" not in str(raised.value)
    # Not chained: a traceback never prints the model's message.
    assert raised.value.__suppress_context__


def test_ocr_misspelled_exams_are_not_names() -> None:
    text = "3. 15H\n4. FERRLTINA\n5. HEMOGRAMA COMPLETQ"

    assert shared_ner_detector().detect(text) == []


# In a whole order the model misses names it finds on the line alone: it reads both.
def test_names_inside_a_multi_line_order_are_found(redactor: PiiRedactor) -> None:
    order = (
        "SOLICITAÇÃO DE EXAMES\n"
        "1. Hemograma completo\n"
        "ANA BEATRIZ MOURA - 42 anos\n"
        "RAFAEL\n"
        "COSTA MENDES - 50 anos\n"
        "2. Ferritina"
    )

    masked = redactor.redact(order).text

    for value in ("ANA", "BEATRIZ", "MOURA", "RAFAEL", "COSTA", "MENDES"):
        assert value not in masked
    assert "1. Hemograma completo" in masked
    assert "2. Ferritina" in masked


def test_a_missing_model_is_an_error_not_a_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import spacy  # noqa: PLC0415

    downloads: list[str] = []
    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    monkeypatch.setattr(spacy.cli, "download", downloads.append)

    with pytest.raises(RuntimeError, match="not installed"):
        NerPersonDetector()
    assert downloads == []
