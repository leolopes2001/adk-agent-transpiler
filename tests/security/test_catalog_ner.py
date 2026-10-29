"""No exam of the catalog is masked by the NER model, in any context or case.

Every name and alias, as written and in upper case, in the contexts of an
order and after person labels, goes through the OCR server's detectors (rules
and NER). Slow (about 20 s), so it runs with the security tests; the same
sweep with the rules alone is in tests/unit/pii/test_catalog_not_masked.py.
"""

import json
from pathlib import Path

import pytest

from pii_guard import PiiEntity, PiiRedactor, document_detectors, shared_ner_detector

pytestmark = pytest.mark.security

CATALOG = Path(__file__).resolve().parents[2] / "data" / "exams_catalog.json"
# The contexts of tests/unit/pii/test_catalog_not_masked.py.
NEUTRAL_CONTEXTS = ["{}", "1. {}", "- {}", "Solicito: {}", "Exames: {}, Hemograma completo"]
LABEL_CONTEXTS = ["Paciente: {}", "Nome: {}", "Dr. {}", "Médico solicitante: {}", "Sra. {}"]
_CATALOG = json.loads(CATALOG.read_text(encoding="utf-8"))
MENTIONS = sorted({m for exam in _CATALOG["exams"] for m in [exam["name"], *exam["aliases"]]})


@pytest.fixture(scope="module")
def redactor() -> PiiRedactor:
    return PiiRedactor(document_detectors())


def _cased(mention: str, case: str) -> str:
    return mention.upper() if case == "upper" else mention


@pytest.mark.parametrize("case", ["as_written", "upper"])
@pytest.mark.parametrize("context", NEUTRAL_CONTEXTS)
def test_no_detector_fires_on_catalog_mentions(
    redactor: PiiRedactor, context: str, case: str
) -> None:
    masked = {
        mention: result.text
        for mention in MENTIONS
        if (result := redactor.redact(context.format(_cased(mention, case)))).spans
    }
    assert masked == {}


# After a label the rule finds the name; the model must not take an exam for one either.
@pytest.mark.parametrize("case", ["as_written", "upper"])
@pytest.mark.parametrize("context", LABEL_CONTEXTS)
def test_ner_never_takes_a_catalog_mention_for_a_name(context: str, case: str) -> None:
    detector = shared_ner_detector()
    as_person = {}
    for mention in MENTIONS:
        text = context.format(_cased(mention, case))
        found = [text[s.start : s.end] for s in detector.detect(text)]
        if found:
            as_person[mention] = found
    assert as_person == {}


def test_whole_catalog_as_one_order_keeps_every_mention(redactor: PiiRedactor) -> None:
    order = "Paciente: Maria Fictícia da Silva\n" + "\n".join(
        f"{n}. {m}" for n, m in enumerate(MENTIONS, start=1)
    )
    result = redactor.redact(order)

    assert result.counts == {PiiEntity.PERSON: 1}
    assert result.lines[1:] == tuple(order.splitlines()[1:])
